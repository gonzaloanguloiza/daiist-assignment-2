"""
Assignment 2: Gradio dashboard.

Run with:  uv run python main.py app

Loads ONLY what train.py already produced: weights/best.pt and artifacts/*.
Nothing is trained here. The decision maths comes from decision.py, the same
module train.py used to pick the operating point.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from decision import (THRESHOLDS, detection_confusion, evaluate, image_scores, outcome_counts,
                      single_threshold_sweep, workload)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"


def load_json(name: str) -> dict:
    return json.loads((ART / name).read_text())


OP = load_json("operating_point.json")
DUMP = load_json("test_predictions.json")
METRICS = load_json("metrics_finetuned.json")
BASELINES = load_json("baselines.json")
DSTATS = load_json("dataset_stats.json")
RUN = load_json("run_info.json") if (ART / "run_info.json").exists() else {}

CLASSES: list[str] = DUMP["classes"]
DEFECT_IDS: set[int] = set(DUMP["defect_class_ids"])
Y, S = image_scores(DUMP["records"], DEFECT_IDS)
COST0 = OP["cost_matrix_eur"]
BASE0 = OP["base_rate"]
CAP = OP.get("capacity", {})
OPT = OP["two_threshold_optimum"]
T_LOW0, T_HIGH0 = OPT["t_low"], OPT["t_high"]

# The model is loaded once, for the upload tab only.
from ultralytics import YOLO
MODEL = YOLO(str(WEIGHTS))


# -----------------------------------------------------------------------------
# Tab 1: upload an image
# -----------------------------------------------------------------------------
def predict(img: Image.Image | None, t_low: float, t_high: float):
    if img is None:
        return None, "Upload an image."
    res = MODEL.predict(img, conf=min(t_low, 0.25), iou=0.6, verbose=False)[0]
    # Draw only boxes at or above the pass/review edge so the picture matches the verdict
    keep = res.boxes.conf >= t_low
    res.boxes = res.boxes[keep]
    plotted = Image.fromarray(res.plot()[:, :, ::-1])
    defect_conf = [float(c) for c, k in zip(res.boxes.conf, res.boxes.cls) if int(k) in DEFECT_IDS]
    score = max(defect_conf, default=0.0)
    if score >= t_high:
        verdict = f"**FAIL - dispatch crew.** Highest defect confidence {score:.2f} >= {t_high:.2f}"
    elif score >= t_low:
        verdict = f"**REVIEW - send to inspector.** Highest defect confidence {score:.2f} in [{t_low:.2f}, {t_high:.2f})"
    else:
        verdict = f"**PASS.** Highest defect confidence {score:.2f} < {t_low:.2f}"
    dets = ", ".join(f"{CLASSES[int(k)]} {float(c):.2f}" for c, k in zip(res.boxes.conf, res.boxes.cls)) or "no boxes above the pass edge"
    return plotted, verdict + f"\n\nDetections: {dets}"


# -----------------------------------------------------------------------------
# Tab 2: operating point
# -----------------------------------------------------------------------------
def cost_dict(fn, fp, tp, tn, review) -> dict:
    return {"fn": float(fn), "fp": float(fp), "tp": float(tp), "tn": float(tn), "review": float(review)}


def operating_point(t_low, t_high, base_rate, fn, fp, tp, tn, review):
    t_low, t_high = float(t_low), float(t_high)
    if t_high < t_low:
        t_high = t_low
    cost = cost_dict(fn, fp, tp, tn, review)
    c = evaluate(Y, S, t_low, t_high, base_rate, cost)
    cnt = outcome_counts(Y, S, t_low, t_high)

    # Confusion matrix: rows = truth, cols = decision. Test counts + per-1,000 at the base rate.
    pi = base_rate
    n_pos, n_neg = max(cnt["n_pos"], 1), max(cnt["n_neg"], 1)
    cm = pd.DataFrame({
        "": ["Defect present (test n=%d)" % cnt["n_pos"], "No defect (test n=%d)" % cnt["n_neg"],
             f"Per 1,000 images at {pi:.0%} base rate: defect", "Per 1,000 images: no defect"],
        "PASS": [cnt["pos_pass"], cnt["neg_pass"],
                 round(1000 * pi * cnt["pos_pass"] / n_pos, 1), round(1000 * (1 - pi) * cnt["neg_pass"] / n_neg, 1)],
        "REVIEW": [cnt["pos_review"], cnt["neg_review"],
                   round(1000 * pi * cnt["pos_review"] / n_pos, 1), round(1000 * (1 - pi) * cnt["neg_review"] / n_neg, 1)],
        "FAIL": [cnt["pos_fail"], cnt["neg_fail"],
                 round(1000 * pi * cnt["pos_fail"] / n_pos, 1), round(1000 * (1 - pi) * cnt["neg_fail"] / n_neg, 1)],
    })
    wl = workload(c["review_share"], CAP.get("images_per_day", 20000), CAP.get("seconds_per_review", 120),
                  CAP.get("inspector_hours_per_day", 7))
    cap_share = CAP.get("max_review_share")
    fits = "" if cap_share is None else (
        f" - fits the {CAP.get('inspectors_available')} inspector(s) available"
        if c["review_share"] <= cap_share else
        f" - EXCEEDS capacity ({CAP.get('inspectors_available')} inspector(s) = {cap_share:.1%} of images)")
    summary = (
        f"### Expected cost: **{c['cost_per_1000']:,.0f} EUR per 1,000 images**\n"
        f"- Missed defects (FN): {c['fn_per_1000']:.1f} per 1,000 - False dispatches (FP): {c['fp_per_1000']:.1f} per 1,000\n"
        f"- Sent to review: **{c['review_share']:.1%}** = {wl['reviewed_per_day']:,.0f} images/day "
        f"= {wl['inspector_hours_per_day']:.1f} inspector-hours = **{wl['inspectors_needed']:.1f} inspectors**{fits}\n"
    )
    alts = OP["alternatives"]
    alt_df = pd.DataFrame([
        {"Policy": "Do nothing (pass everything)", "EUR / 1,000": alts["do_nothing"]["cost_per_1000"], "Review share": 0.0},
        {"Policy": "Human checks everything", "EUR / 1,000": alts["human_checks_everything"]["cost_per_1000"], "Review share": 1.0},
        {"Policy": f"Model alone, t={alts['model_alone']['threshold']:.2f}", "EUR / 1,000": alts["model_alone"]["cost_per_1000"], "Review share": 0.0},
        {"Policy": f"Model + review (chosen) [{T_LOW0:.2f}, {T_HIGH0:.2f})", "EUR / 1,000": OPT["cost_per_1000"], "Review share": OPT["review_share"]},
        {"Policy": f"Your sliders [{t_low:.2f}, {t_high:.2f})", "EUR / 1,000": c["cost_per_1000"], "Review share": c["review_share"]},
    ])
    return summary, cm, alt_df, cost_curve(t_low, t_high, base_rate, cost)


def cost_curve(t_low, t_high, base_rate, cost):
    rows = single_threshold_sweep(Y, S, base_rate, cost)
    xs = [r["threshold"] for r in rows]; ys = [r["cost_per_1000"] for r in rows]
    best = min(rows, key=lambda r: r["cost_per_1000"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xs, ys, label="single threshold, no review")
    ax.scatter([best["threshold"]], [best["cost_per_1000"]], c="C3", zorder=5,
               label=f"min at t={best['threshold']:.2f} ({best['cost_per_1000']:,.0f} EUR)")
    ax.axvline(T_HIGH0, ls="--", c="C3", alpha=0.6, label=f"chosen fail edge t_high={T_HIGH0:.2f}")
    ax.axvspan(t_low, t_high, alpha=0.15, color="C1", label=f"current review band [{t_low:.2f}, {t_high:.2f})")
    two = evaluate(Y, S, t_low, t_high, base_rate, cost)["cost_per_1000"]
    ax.axhline(two, ls=":", c="C1", label=f"with review band: {two:,.0f} EUR")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("EUR per 1,000 images")
    ax.set_title(f"Cost vs threshold at base rate {base_rate:.1%}")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Tab 3: per-class results
# -----------------------------------------------------------------------------
def per_class_table() -> pd.DataFrame:
    rows = []
    for name, m in METRICS["per_class"].items():
        rows.append({"Class": name, "Role": "DEFECT (positive)" if CLASSES.index(name) in DEFECT_IDS else "good (negative)",
                     "Test instances": DSTATS["splits"]["test"]["instances_per_class"].get(name),
                     "Precision": m["precision"], "Recall": m["recall"], "AP50": m["ap50"], "AP50-95": m["ap50_95"]})
    rows.append({"Class": "ALL (mAP)", "Role": "", "Test instances": sum(DSTATS["splits"]["test"]["instances_per_class"].values()),
                 "Precision": None, "Recall": None, "AP50": METRICS["map50"], "AP50-95": METRICS["map50_95"]})
    return pd.DataFrame(rows)


def ultralytics_confusion() -> pd.DataFrame:
    cm = METRICS["confusion_matrix"]
    labels = cm["labels"]
    df = pd.DataFrame(cm["matrix"], index=[f"pred: {l}" for l in labels], columns=[f"true: {l}" for l in labels])
    return df.reset_index().rename(columns={"index": ""})


def detection_confusion_at(conf: float) -> pd.DataFrame:
    m = detection_confusion(DUMP["records"], len(CLASSES), conf)
    labels = CLASSES + ["background"]
    df = pd.DataFrame(m, index=[f"pred: {l}" for l in labels], columns=[f"true: {l}" for l in labels])
    return df.reset_index().rename(columns={"index": ""})


def baselines_table() -> pd.DataFrame:
    b = BASELINES
    def row(name, blk, key):
        d = blk[key]
        return {"Model": name, "mAP50": blk["map50"], "mAP50-95": blk["map50_95"],
                "Precision (defects, IoU.5)": d["precision"], "Recall (defects, IoU.5)": d["recall"], "F1": d["f1"],
                "TP": d["tp"], "FP": d["fp"], "FN": d["fn"]}
    return pd.DataFrame([
        row("Finetuned yolo11n", b["finetuned"], "defects_only"),
        row("COCO yolo11n zero-shot (class-agnostic)", b["coco_zero_shot"], "class_agnostic_defects"),
        row("Predict nothing", b["predict_nothing"], "defects_only"),
    ])


def dataset_table() -> pd.DataFrame:
    rows = []
    for split, s in DSTATS["splits"].items():
        rows.append({"Split": split, "Images": s["images"], "Images with defect": s["images_with_defect"],
                     "Positive share": s["positive_share"], **s["instances_per_class"],
                     "Median box / image area": s["box_area_share_of_image"]["median"]})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
with gr.Blocks(title="Track fastener inspection - operating point") as demo:
    gr.Markdown(
        f"# Rail fastener inspection: pass / review / fail\n"
        f"Finetuned `{RUN.get('base_model', 'yolo11n')}` on `{RUN.get('dataset', '')}` "
        f"({RUN.get('epochs', '?')} epochs, {RUN.get('device', '?')}). "
        f"Decision unit: one track image. Positive = any of {[CLASSES[i] for i in sorted(DEFECT_IDS)]}. "
        f"Score = highest defect confidence in the image."
    )
    with gr.Row():
        t_low = gr.Slider(0.0, 1.0, value=T_LOW0, step=0.01, label="t_low: pass below / review at or above")
        t_high = gr.Slider(0.0, 1.0, value=T_HIGH0, step=0.01, label="t_high: fail at or above")

    with gr.Tab("Upload an image"):
        with gr.Row():
            inp = gr.Image(type="pil", label="Track image")
            out_img = gr.Image(label="Predictions (boxes at or above t_low)")
        verdict = gr.Markdown()
        inp.change(predict, [inp, t_low, t_high], [out_img, verdict])
        t_low.release(predict, [inp, t_low, t_high], [out_img, verdict])
        t_high.release(predict, [inp, t_low, t_high], [out_img, verdict])

    with gr.Tab("Operating point"):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.5, value=BASE0, step=0.005, label="Assumed base rate (share of images with a defect)")
        with gr.Row():
            c_fn = gr.Number(COST0["fn"], label="Cost of a missed defect (FN), EUR")
            c_fp = gr.Number(COST0["fp"], label="Cost of a false dispatch (FP), EUR")
            c_rev = gr.Number(COST0["review"], label="Cost of one human review, EUR")
            c_tp = gr.Number(COST0["tp"], label="Extra cost of a correct dispatch (TP), EUR")
            c_tn = gr.Number(COST0["tn"], label="Cost of a correct pass (TN), EUR")
        summary = gr.Markdown()
        with gr.Row():
            cm_df = gr.Dataframe(label="Confusion matrix: truth x decision (test counts, then per 1,000 at the base rate)", interactive=False)
            alt_df = gr.Dataframe(label="Priced against the alternatives", interactive=False)
        curve = gr.Plot(label="Cost against threshold")
        inputs = [t_low, t_high, base_rate, c_fn, c_fp, c_tp, c_tn, c_rev]
        outputs = [summary, cm_df, alt_df, curve]
        for comp in inputs:
            comp.change(operating_point, inputs, outputs)
        demo.load(operating_point, inputs, outputs)

    with gr.Tab("Per-class results"):
        gr.Markdown("Detection-level metrics on the test split (ultralytics `val`, IoU 0.5 for AP50).")
        gr.Dataframe(per_class_table(), interactive=False, label="Per class")
        gr.Dataframe(baselines_table(), interactive=False, label="Finetuned vs baselines, defect classes, conf 0.25")
        gr.Dataframe(detection_confusion_at(0.25), interactive=False,
                     label="Detection confusion matrix at conf 0.25, IoU 0.5 (rows predicted, columns true, last = background)")
        gr.Markdown("The ultralytics matrix below is accumulated at the validator's low confidence floor, "
                    "so its background column counts every low-confidence box; it is shown for completeness.")
        gr.Dataframe(ultralytics_confusion(), interactive=False, label="Ultralytics confusion matrix (val defaults)")
        if (ART / "confusion_matrix.png").exists():
            gr.Image(str(ART / "confusion_matrix.png"), label="Ultralytics confusion matrix plot", interactive=False)

    with gr.Tab("Dataset"):
        gr.Dataframe(dataset_table(), interactive=False, label="Splits and instances per class")
        if (ART / "results.png").exists():
            gr.Image(str(ART / "results.png"), label="Training curves", interactive=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
