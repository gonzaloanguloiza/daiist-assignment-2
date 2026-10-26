"""
The decision layer, shared by train.py (which picks the operating point) and
app.py (which lets you move it). One definition, so the dashboard and the
report can never disagree.

Decision unit: one track image. Positive = contains at least one defect
instance. Score = highest confidence among the image's defect detections.

    score <  t_low            -> pass   (no action)
    t_low <= score < t_high   -> review (an inspector looks at it)
    score >= t_high           -> fail   (a maintenance crew is sent)

Everything is re-weighted from the test split's class mix to an assumed
production base rate, because a test split with 44% positives says nothing
about cost per 1,000 real images.
"""

from __future__ import annotations

import numpy as np

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def image_scores(records: list[dict], defect_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, score) per image from the dumped test predictions."""
    y = np.array([any(g["cls"] in defect_ids for g in r["gt"]) for r in records], dtype=bool)
    s = np.array([max((p["conf"] for p in r["pred"] if p["cls"] in defect_ids), default=0.0) for r in records])
    return y, s


def outcome_counts(y: np.ndarray, s: np.ndarray, t_low: float, t_high: float) -> dict:
    """Raw counts on the evaluated images: {pos|neg}_{pass|review|fail}."""
    fail = s >= t_high
    review = (s >= t_low) & ~fail
    pas = s < t_low
    return {
        "pos_pass": int((y & pas).sum()), "pos_review": int((y & review).sum()), "pos_fail": int((y & fail).sum()),
        "neg_pass": int((~y & pas).sum()), "neg_review": int((~y & review).sum()), "neg_fail": int((~y & fail).sum()),
        "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
    }


def rates(y: np.ndarray, s: np.ndarray, t_low: float, t_high: float) -> dict:
    c = outcome_counts(y, s, t_low, t_high)
    n_pos, n_neg = max(c["n_pos"], 1), max(c["n_neg"], 1)
    return {k: c[k] / (n_pos if k.startswith("pos") else n_neg)
            for k in ("pos_pass", "pos_review", "pos_fail", "neg_pass", "neg_review", "neg_fail")}


def cost_per_1000(r: dict, base_rate: float, cost: dict) -> dict:
    """Expected cost per 1,000 images at the assumed base rate.

    A reviewed image is resolved correctly by the inspector, so a reviewed
    positive costs review + tp and a reviewed negative costs review + tn.
    """
    pi = base_rate
    c_pos = r["pos_pass"] * cost["fn"] + r["pos_review"] * (cost["review"] + cost["tp"]) + r["pos_fail"] * cost["tp"]
    c_neg = r["neg_pass"] * cost["tn"] + r["neg_review"] * (cost["review"] + cost["tn"]) + r["neg_fail"] * cost["fp"]
    total = 1000 * (pi * c_pos + (1 - pi) * c_neg)
    review_share = pi * r["pos_review"] + (1 - pi) * r["neg_review"]
    return {
        "cost_per_1000": round(float(total), 2),
        "review_share": round(float(review_share), 4),
        "fn_per_1000": round(float(1000 * pi * r["pos_pass"]), 2),
        "fp_per_1000": round(float(1000 * (1 - pi) * r["neg_fail"]), 2),
        "tp_per_1000": round(float(1000 * pi * r["pos_fail"]), 2),
        "tn_per_1000": round(float(1000 * (1 - pi) * r["neg_pass"]), 2),
    }


def evaluate(y, s, t_low, t_high, base_rate, cost) -> dict:
    return cost_per_1000(rates(y, s, t_low, t_high), base_rate, cost)


def single_threshold_sweep(y, s, base_rate, cost) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        r = rates(y, s, t, t)
        rows.append({"threshold": float(t), **cost_per_1000(r, base_rate, cost),
                     "tpr": round(float(r["pos_fail"]), 4), "fpr": round(float(r["neg_fail"]), 4)})
    return rows


def two_threshold_optimum(y, s, base_rate, cost, max_review_share: float = 1.0) -> dict:
    """Cheapest (t_low, t_high) whose review share fits the capacity cap."""
    best = None
    for i, t_low in enumerate(THRESHOLDS):
        for t_high in THRESHOLDS[i:]:
            c = cost_per_1000(rates(y, s, t_low, t_high), base_rate, cost)
            if c["review_share"] > max_review_share:
                continue
            if best is None or c["cost_per_1000"] < best["cost_per_1000"]:
                best = {"t_low": float(t_low), "t_high": float(t_high), **c}
    return best


def workload(review_share: float, images_per_day: int, seconds_per_review: int, inspector_hours: float) -> dict:
    reviewed = images_per_day * review_share
    hours = reviewed * seconds_per_review / 3600
    return {"images_per_day": images_per_day, "reviewed_per_day": round(reviewed, 1),
            "inspector_hours_per_day": round(hours, 1),
            "inspectors_needed": round(hours / inspector_hours, 2)}


# -----------------------------------------------------------------------------
# Detection-level confusion matrix at a chosen confidence (for the dashboard)
# -----------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def detection_confusion(records: list[dict], n_classes: int, conf: float, iou_thr: float = 0.5) -> np.ndarray:
    """(n_classes+1) x (n_classes+1) matrix, rows = predicted, cols = true, last = background.

    Each prediction above `conf` is matched greedily (highest confidence first)
    to the unmatched GT box it overlaps most; unmatched predictions count
    against background, unmatched GT boxes count as background predictions.
    """
    m = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for rec in records:
        gt = rec["gt"]
        pr = sorted((p for p in rec["pred"] if p["conf"] >= conf), key=lambda p: -p["conf"])
        g_boxes = np.array([g["xyxy"] for g in gt]).reshape(-1, 4)
        p_boxes = np.array([p["xyxy"] for p in pr]).reshape(-1, 4)
        ious = iou_matrix(p_boxes, g_boxes)
        matched = set()
        for i, p in enumerate(pr):
            j = -1
            if ious.shape[1]:
                order = np.argsort(-ious[i])
                for cand in order:
                    if cand not in matched and ious[i, cand] >= iou_thr:
                        j = int(cand); break
            if j >= 0:
                matched.add(j); m[p["cls"], gt[j]["cls"]] += 1
            else:
                m[p["cls"], n_classes] += 1
        for j, g in enumerate(gt):
            if j not in matched:
                m[n_classes, g["cls"]] += 1
    return m
