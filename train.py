"""
Assignment 2: training pipeline.

Run with:  uv run python main.py train

End to end, no manual steps:
  1. Download the Roboflow Universe dataset (key via api_key.load_api_key).
  2. Describe the dataset from its labels (instances per class, split sizes,
     object size relative to the image) -> artifacts/dataset_stats.json
  3. Finetune yolo11n from the COCO checkpoint -> weights/best.pt
  4. Evaluate on the TEST split: finetuned model (per-class P/R/AP50, mAP50-95,
     confusion matrix), the COCO model zero-shot, and a predict-nothing baseline.
  5. Dump every test prediction (artifacts/test_predictions.json) so the cost
     sweep and the dashboard can recompute anything without the model.
  6. Sweep the confidence threshold, cost each point with the cost matrix in
     CONFIG, pick the single-threshold optimum and the two-threshold
     (pass / human review / fail) optimum -> artifacts/cost_sweep.csv,
     artifacts/operating_point.json, artifacts/cost_vs_threshold.png

Decision unit: one TRACK IMAGE (a segment of rail photographed by the
inspection camera). The image is "positive" if it contains at least one defect
instance (any class in DEFECT_CLASSES). The model's score for an image is the
highest confidence among its defect detections. A dispatcher acts on it:
  pass  -> no action
  fail  -> send a maintenance crew
  review-> an inspector looks at the image first

Per-class detection metrics are reported as well, because the image-level
decision is only as good as the detections underneath it.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from api_key import load_api_key

# =============================================================================
# CONFIG - every number here must be justified in REPORT.md
# =============================================================================
ROBOFLOW_WORKSPACE = "railway-defects-osolv"
ROBOFLOW_PROJECT = "fastener-detection-7a3vn"
ROBOFLOW_VERSION = 1

BASE_MODEL = "yolo11n.pt"          # pretrained COCO checkpoint we finetune from
EPOCHS = int(os.environ.get("EPOCHS", "40"))   # EPOCHS=1 for a smoke test
IMGSZ = 640
BATCH = 16
SEED = 42

# Which classes mean "there is a problem on this stretch of track".
# Matched by substring, case-insensitive, against the dataset's class names.
DEFECT_KEYWORDS = ("missing", "defective")

# Cost matrix, in euros, per track image. Placeholders: set these yourself.
COST = {
    "fn": 2000.0,   # defect present, image passed: risk-weighted cost of a
                    #   missing fastener left in service until next inspection
    "fp": 150.0,    # no defect, image failed: a crew drives out for nothing
    "tp": 0.0,      # defect present and crew sent: the repair happens anyway,
                    #   so no *extra* cost from the decision
    "tn": 0.0,      # nothing there, nothing done
    "review": 5.0,  # an inspector looks at one image (~2 min of their time)
}
# How often a track image really contains a defect in production. The test
# split is NOT representative of this, so every cost is re-weighted by it.
BASE_RATE = 0.05

# Human review capacity, for the "is it realistic" check in REPORT.md
IMAGES_PER_DAY = 20000      # images the inspection train produces per day
SECONDS_PER_REVIEW = 120    # time an inspector needs per image
INSPECTOR_HOURS_PER_DAY = 7
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / ROBOFLOW_PROJECT
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"
ART.mkdir(exist_ok=True)
WEIGHTS.mkdir(exist_ok=True)

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def log(msg: str) -> None:
    print(f"\n[train] {msg}", flush=True)


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------
# 1. Download
# -----------------------------------------------------------------------------
def download_dataset() -> Path:
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        log(f"dataset already at {DATA_DIR}, skipping download")
    else:
        from roboflow import Roboflow
        log(f"downloading {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT).version(ROBOFLOW_VERSION)
        version.download("yolov11", location=str(DATA_DIR))

    # Make the yaml point at absolute paths so ultralytics never guesses.
    cfg = yaml.safe_load(yaml_path.read_text())
    for split in ("train", "val", "test"):
        folder = DATA_DIR / ("valid" if split == "val" else split) / "images"
        if folder.exists():
            cfg[split] = str(folder)
    cfg["path"] = str(DATA_DIR)
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return yaml_path


# -----------------------------------------------------------------------------
# 2. Describe
# -----------------------------------------------------------------------------
def parse_label_line(line: str) -> list[float]:
    """YOLO label line -> [cls, cx, cy, w, h].

    Roboflow exports some instances as polygons (cls x1 y1 x2 y2 ...). Ultralytics
    turns those into boxes for a detect task; we do the same here so the stats
    and the GT used for costing match what the model was trained on.
    """
    vals = list(map(float, line.split()))
    if len(vals) == 5:
        return vals
    pts = np.array(vals[1:], dtype=float).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [vals[0], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def read_labels(label_dir: Path) -> dict[str, np.ndarray]:
    """{image_stem: array of [cls, cx, cy, w, h]} for one split."""
    out = {}
    for p in sorted(label_dir.glob("*.txt")):
        rows = [parse_label_line(line) for line in p.read_text().splitlines() if line.strip()]
        out[p.stem] = np.array(rows, dtype=float).reshape(-1, 5)
    return out


def describe_dataset(yaml_path: Path, names: list[str], defect_ids: set[int]) -> dict:
    stats = {"classes": names, "defect_class_ids": sorted(defect_ids), "splits": {}}
    for split in ("train", "valid", "test"):
        lbl_dir = DATA_DIR / split / "labels"
        img_dir = DATA_DIR / split / "images"
        if not lbl_dir.exists():
            continue
        labels = read_labels(lbl_dir)
        n_images = len(list(img_dir.glob("*.*")))
        per_class = {n: 0 for n in names}
        areas: list[float] = []
        positives = 0
        for rows in labels.values():
            if rows.size and any(int(c) in defect_ids for c in rows[:, 0]):
                positives += 1
            for c, _, _, w, h in rows:
                per_class[names[int(c)]] += 1
                areas.append(float(w * h))
        areas_arr = np.array(areas) if areas else np.zeros(1)
        stats["splits"][split] = {
            "images": n_images,
            "labelled_images": len(labels),
            "images_with_defect": positives,
            "positive_share": round(positives / max(len(labels), 1), 4),
            "instances_per_class": per_class,
            "box_area_share_of_image": {
                "median": round(float(np.median(areas_arr)), 5),
                "p10": round(float(np.percentile(areas_arr, 10)), 5),
                "p90": round(float(np.percentile(areas_arr, 90)), 5),
            },
        }
    (ART / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    log("dataset stats:\n" + json.dumps(stats["splits"], indent=2))
    return stats


# -----------------------------------------------------------------------------
# 3. Finetune
# -----------------------------------------------------------------------------
def finetune(yaml_path: Path, device: str) -> Path:
    from ultralytics import YOLO
    log(f"finetuning {BASE_MODEL} for {EPOCHS} epochs on device={device}")
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
        device=device, project=str(RUNS), name="finetune", exist_ok=True,
        workers=0 if device == "mps" else 4, plots=True, verbose=True,
    )
    best = RUNS / "finetune" / "weights" / "best.pt"
    target = WEIGHTS / "best.pt"
    shutil.copy2(best, target)
    log(f"best weights copied to {target}")
    return target


# -----------------------------------------------------------------------------
# 4. Evaluate: finetuned per-class + confusion matrix
# -----------------------------------------------------------------------------
def evaluate_finetuned(weights: Path, yaml_path: Path, names: list[str], device: str) -> dict:
    from ultralytics import YOLO
    log("evaluating finetuned model on the test split")
    model = YOLO(str(weights))
    r = model.val(data=str(yaml_path), split="test", device=device, plots=True,
                  project=str(RUNS), name="val_test", exist_ok=True, verbose=False)
    per_class = {}
    for i, cid in enumerate(r.box.ap_class_index):
        per_class[names[int(cid)]] = {
            "precision": round(float(r.box.p[i]), 4),
            "recall": round(float(r.box.r[i]), 4),
            "ap50": round(float(r.box.ap50[i]), 4),
            "ap50_95": round(float(r.box.ap[i]), 4),
        }
    cm = r.confusion_matrix.matrix.astype(int).tolist()  # rows=pred, cols=true, last=background
    out = {
        "map50": round(float(r.box.map50), 4),
        "map50_95": round(float(r.box.map), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": names + ["background"], "rows_are_predicted": True, "matrix": cm},
    }
    (ART / "metrics_finetuned.json").write_text(json.dumps(out, indent=2))
    log(f"finetuned: mAP50={out['map50']} mAP50-95={out['map50_95']}")
    return out


# -----------------------------------------------------------------------------
# 5. Dump predictions + class-agnostic matching (for baselines table)
# -----------------------------------------------------------------------------
def xywhn_to_xyxy(rows: np.ndarray, w: int, h: int) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((0, 4))
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def predict_test(weights: Path, device: str, conf: float = 0.001) -> list[dict]:
    """One record per test image with GT boxes and all predicted boxes."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    img_dir = DATA_DIR / "test" / "images"
    lbl_dir = DATA_DIR / "test" / "labels"
    labels = read_labels(lbl_dir)
    records = []
    images = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for res in model.predict(images, conf=conf, iou=0.6, max_det=300, imgsz=IMGSZ,
                             device=device, stream=True, verbose=False):
        stem = Path(res.path).stem
        h, w = res.orig_shape
        gt = labels.get(stem, np.zeros((0, 5)))
        records.append({
            "image": Path(res.path).name,
            "gt": [{"cls": int(c), "xyxy": [round(float(v), 1) for v in box]}
                   for (c, *_), box in zip(gt, xywhn_to_xyxy(gt, w, h))],
            "pred": [{"cls": int(c), "conf": round(float(s), 4), "xyxy": [round(float(v), 1) for v in box]}
                     for c, s, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())],
        })
    return records


def detection_stats(records: list[dict], conf: float, defect_ids: set[int] | None, class_agnostic: bool) -> dict:
    """Precision/recall at IoU 0.5 for boxes above `conf`.

    defect_ids=None -> all classes count. class_agnostic -> ignore predicted
    class (needed for the COCO zero-shot model, whose classes are not ours).
    """
    tp = fp = fn = 0
    for rec in records:
        gt = [g for g in rec["gt"] if defect_ids is None or g["cls"] in defect_ids]
        pr = [p for p in rec["pred"] if p["conf"] >= conf and
              (class_agnostic or defect_ids is None or p["cls"] in defect_ids)]
        pr.sort(key=lambda p: -p["conf"])
        g_boxes = np.array([g["xyxy"] for g in gt]).reshape(-1, 4)
        p_boxes = np.array([p["xyxy"] for p in pr]).reshape(-1, 4)
        ious = iou_matrix(p_boxes, g_boxes)
        matched = set()
        for i in range(len(pr)):
            j = int(ious[i].argmax()) if ious.shape[1] else -1
            if j >= 0 and ious[i, j] >= 0.5 and j not in matched and \
                    (class_agnostic or pr[i]["cls"] == gt[j]["cls"]):
                matched.add(j); tp += 1
            else:
                fp += 1
        fn += len(gt) - len(matched)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec_, 4),
            "f1": round(2 * prec * rec_ / (prec + rec_), 4) if prec + rec_ else 0.0}


def evaluate_baselines(ft_records: list[dict], device: str, defect_ids: set[int], finetuned_metrics: dict) -> dict:
    log("evaluating COCO zero-shot and predict-nothing baselines on the test split")
    coco_records = predict_test(ROOT / BASE_MODEL, device)
    conf = 0.25
    table = {
        "evaluated_at_conf": conf, "iou": 0.5,
        "finetuned": {
            "map50": finetuned_metrics["map50"], "map50_95": finetuned_metrics["map50_95"],
            "all_classes": detection_stats(ft_records, conf, None, class_agnostic=False),
            "defects_only": detection_stats(ft_records, conf, defect_ids, class_agnostic=False),
        },
        "coco_zero_shot": {
            "map50": 0.0, "map50_95": 0.0,
            "note": "COCO has none of our classes, so per-class AP is 0 by construction; "
                    "class-agnostic box overlap is the fairest thing we can report.",
            "class_agnostic_any_box": detection_stats(coco_records, conf, None, class_agnostic=True),
            "class_agnostic_defects": detection_stats(coco_records, conf, defect_ids, class_agnostic=True),
        },
        "predict_nothing": {
            "map50": 0.0, "map50_95": 0.0,
            "all_classes": detection_stats([{"gt": r["gt"], "pred": []} for r in ft_records], conf, None, False),
            "defects_only": detection_stats([{"gt": r["gt"], "pred": []} for r in ft_records], conf, defect_ids, False),
        },
    }
    (ART / "baselines.json").write_text(json.dumps(table, indent=2))
    return table


# -----------------------------------------------------------------------------
# 6. Cost sweep (image-level decision)
# -----------------------------------------------------------------------------
def image_scores(records: list[dict], defect_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([any(g["cls"] in defect_ids for g in r["gt"]) for r in records], dtype=bool)
    s = np.array([max((p["conf"] for p in r["pred"] if p["cls"] in defect_ids), default=0.0) for r in records])
    return y, s


def rates(y: np.ndarray, s: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of positives / negatives landing in pass, review, fail."""
    pos, neg = y, ~y
    fail = s >= t_high
    review = (s >= t_low) & ~fail
    pas = s < t_low
    n_pos, n_neg = max(pos.sum(), 1), max(neg.sum(), 1)
    return {
        "pos_pass": (pos & pas).sum() / n_pos, "pos_review": (pos & review).sum() / n_pos, "pos_fail": (pos & fail).sum() / n_pos,
        "neg_pass": (neg & pas).sum() / n_neg, "neg_review": (neg & review).sum() / n_neg, "neg_fail": (neg & fail).sum() / n_neg,
    }


def cost_per_1000(r: dict, base_rate: float = BASE_RATE, cost: dict = COST) -> dict:
    """Expected cost per 1,000 images at the assumed base rate.

    Reviewed images are resolved correctly by the inspector, so a reviewed
    positive costs review + tp, a reviewed negative costs review + tn.
    """
    pi = base_rate
    c_pos = r["pos_pass"] * cost["fn"] + r["pos_review"] * (cost["review"] + cost["tp"]) + r["pos_fail"] * cost["tp"]
    c_neg = r["neg_pass"] * cost["tn"] + r["neg_review"] * (cost["review"] + cost["tn"]) + r["neg_fail"] * cost["fp"]
    total = 1000 * (pi * c_pos + (1 - pi) * c_neg)
    review_share = pi * r["pos_review"] + (1 - pi) * r["neg_review"]
    return {"cost_per_1000": round(float(total), 2), "review_share": round(float(review_share), 4),
            "fn_per_1000": round(float(1000 * pi * r["pos_pass"]), 2),
            "fp_per_1000": round(float(1000 * (1 - pi) * r["neg_fail"]), 2)}


def cost_sweep(y: np.ndarray, s: np.ndarray) -> dict:
    # Single threshold: fail if score >= t, else pass. No review.
    rows = []
    for t in THRESHOLDS:
        r = rates(y, s, t, t)
        c = cost_per_1000(r)
        rows.append({"threshold": float(t), **c, "tpr": round(float(r["pos_fail"]), 4), "fpr": round(float(r["neg_fail"]), 4)})
    with (ART / "cost_sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    best_single = min(rows, key=lambda r: r["cost_per_1000"])

    # Two thresholds: pass < t_low <= review < t_high <= fail.
    best_two = None
    for i, t_low in enumerate(THRESHOLDS):
        for t_high in THRESHOLDS[i:]:
            c = cost_per_1000(rates(y, s, t_low, t_high))
            if best_two is None or c["cost_per_1000"] < best_two["cost_per_1000"]:
                best_two = {"t_low": float(t_low), "t_high": float(t_high), **c}

    # Alternatives the brief asks us to price against
    do_nothing = cost_per_1000(rates(y, s, 1.01, 1.01))          # everything passes
    check_everything = cost_per_1000(rates(y, s, 0.0, 1.01))     # everything reviewed
    model_alone = {"threshold": best_single["threshold"], **{k: best_single[k] for k in ("cost_per_1000", "fn_per_1000", "fp_per_1000")}}

    # Review workload at the two-threshold optimum
    images_reviewed = IMAGES_PER_DAY * best_two["review_share"]
    hours = images_reviewed * SECONDS_PER_REVIEW / 3600
    workload = {"images_per_day": IMAGES_PER_DAY, "reviewed_per_day": round(images_reviewed, 1),
                "inspector_hours_per_day": round(hours, 1),
                "inspectors_needed": round(hours / INSPECTOR_HOURS_PER_DAY, 2)}

    op = {
        "decision_unit": "track image; positive = contains >=1 defect instance; score = max defect confidence",
        "cost_matrix_eur": COST, "base_rate": BASE_RATE,
        "test_split_positive_share": round(float(y.mean()), 4), "test_images": int(len(y)),
        "single_threshold_optimum": best_single,
        "two_threshold_optimum": best_two,
        "alternatives": {"do_nothing": do_nothing, "human_checks_everything": check_everything, "model_alone": model_alone},
        "review_workload": workload,
    }
    (ART / "operating_point.json").write_text(json.dumps(op, indent=2))
    log("operating point:\n" + json.dumps(op, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["threshold"] for r in rows], [r["cost_per_1000"] for r in rows], label="single threshold (no review)")
    ax.axvline(best_single["threshold"], ls="--", c="C3", label=f"chosen t={best_single['threshold']:.2f}")
    ax.axvspan(best_two["t_low"], best_two["t_high"], alpha=0.15, color="C1",
               label=f"review band [{best_two['t_low']:.2f}, {best_two['t_high']:.2f})")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("expected cost per 1,000 images (EUR)")
    ax.set_title(f"Cost vs threshold (base rate {BASE_RATE:.0%}, FN={COST['fn']:.0f} EUR, FP={COST['fp']:.0f} EUR)")
    ax.legend(); fig.tight_layout(); fig.savefig(ART / "cost_vs_threshold.png", dpi=150)
    return op


# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    device = pick_device()
    yaml_path = download_dataset()
    names = list(yaml.safe_load(yaml_path.read_text())["names"])
    defect_ids = {i for i, n in enumerate(names) if any(k in n.lower() for k in DEFECT_KEYWORDS)}
    log(f"classes={names}  defect classes={[names[i] for i in sorted(defect_ids)]}")
    if not defect_ids:
        raise SystemExit("No class matched DEFECT_KEYWORDS; fix CONFIG.")

    describe_dataset(yaml_path, names, defect_ids)
    weights = finetune(yaml_path, device)
    ft_metrics = evaluate_finetuned(weights, yaml_path, names, device)

    records = predict_test(weights, device)
    (ART / "test_predictions.json").write_text(json.dumps(
        {"classes": names, "defect_class_ids": sorted(defect_ids), "records": records}))
    evaluate_baselines(records, device, defect_ids, ft_metrics)

    y, s = image_scores(records, defect_ids)
    cost_sweep(y, s)

    # Copy the useful training plots next to the other artifacts
    for name in ("results.png", "confusion_matrix.png"):
        src = RUNS / "val_test" / name if name.startswith("confusion") else RUNS / "finetune" / name
        if src.exists():
            shutil.copy2(src, ART / name)

    import torch, ultralytics
    (ART / "run_info.json").write_text(json.dumps({
        "dataset": f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/v{ROBOFLOW_VERSION}",
        "base_model": BASE_MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
        "device": device, "machine": platform.platform(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))
    log(f"done in {(time.time() - t0) / 60:.1f} min. Artifacts in {ART}, weights in {WEIGHTS}")


if __name__ == "__main__":
    main()
