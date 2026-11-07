"""Shared metric computation — used by both train.py and app.py so the
numbers in the README and the numbers the app shows on test_data.csv come
from the exact same code path.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

AVERAGING = "macro"  # multiclass, heavily imbalanced — macro so rare species count


def compute_metrics(y_true, y_pred, y_proba=None, class_labels=None) -> dict:
    """y_proba, if given, must have columns ordered to match class_labels
    (i.e. model.classes_) — that's what predict_proba returns.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=AVERAGING, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=AVERAGING, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=AVERAGING, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is None:
        metrics["auc"] = "N/A (no probabilities)"
        return metrics

    try:
        auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average=AVERAGING,
            labels=class_labels if class_labels is not None else np.unique(y_true),
        )
        # when a class the model was trained on never shows up in y_true,
        # sklearn doesn't always raise — depending on version it can warn
        # and hand back nan for that class's contribution to the macro
        # average instead. Either way, it means the same thing here.
        metrics["auc"] = "N/A (class missing)" if np.isnan(auc) else auc
    except ValueError:
        metrics["auc"] = "N/A (class missing)"

    return metrics
