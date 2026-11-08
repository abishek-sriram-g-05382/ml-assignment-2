# Import functions for computing metrics from sklearn, and numpy for numerical operations
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

# Function to compute various classification metrics
def compute_metrics(y_true, y_pred, y_proba=None, class_labels=None) -> dict:
    """
    Function to compute classification metrics :
      accuracy, precision, recall, F1 score, 
      Matthews correlation coefficient (MCC), and AUC (Area Under the Curve).
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
        metrics["auc"] = "N/A (class missing)" if np.isnan(auc) else auc
    except ValueError:
        metrics["auc"] = "N/A (class missing)"

    return metrics
