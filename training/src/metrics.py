"""Metric helpers used after training.

The metrics are computed with the real ECG labels, so the results stay readable
as -1 and 1 instead of using the model's internal 0/1 ids.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> dict:
    labels = sorted(int(label) for label in np.unique(np.concatenate([y_true, y_pred])))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
