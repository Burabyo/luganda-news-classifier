"""Shared evaluation: every experiment reports the same metrics and writes the same JSON format."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from data import LABELS, RESULTS_DIR


def compute(y_true, y_pred) -> dict:
    """Macro-F1 is the primary metric: the classes are imbalanced (politics = 45% of train),
    so accuracy alone would reward a model that over-predicts politics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def full_report(y_true, y_pred) -> dict:
    return {
        "per_class": classification_report(y_true, y_pred, labels=list(range(len(LABELS))),
                                           target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


def save_run(name: str, record: dict, preds: np.ndarray | None = None, probs: np.ndarray | None = None):
    """Write results/runs/<name>.json and, optionally, test-set predictions for error analysis."""
    out = RESULTS_DIR / "runs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.json", "w") as f:
        json.dump(record, f, indent=2, default=float)
    if preds is not None:
        pred_dir = RESULTS_DIR / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / f"{name}_pred.npy", preds)
        if probs is not None:
            np.save(pred_dir / f"{name}_probs.npy", probs)
