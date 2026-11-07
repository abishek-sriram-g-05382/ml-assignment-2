"""CI smoke test: load each saved model, score test_data.csv, and confirm
the result matches results/metrics_comparison.csv — catches a model/sklearn
version mismatch before Streamlit Cloud does.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd

from src.metrics import compute_metrics

MODEL_NAMES = ["logistic_regression", "decision_tree", "knn", "naive_bayes", "svm"]
TOLERANCE = 1e-9


def main() -> int:
    df = pd.read_csv(REPO_ROOT / "test_data.csv")
    feature_cols = [c for c in df.columns if c != "Species"]
    X, y_true = df[feature_cols], df["Species"]
    expected = pd.read_csv(REPO_ROOT / "results" / "metrics_comparison.csv", index_col=0)

    failures = []
    for name in MODEL_NAMES:
        model_path = REPO_ROOT / "model" / f"{name}.joblib"
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        metrics = compute_metrics(y_true, y_pred, y_proba, class_labels=model.classes_)

        row = expected.loc[name]
        for metric in ["accuracy", "precision", "recall", "f1", "mcc", "auc"]:
            actual, expected_val = float(metrics[metric]), float(row[metric])
            if abs(actual - expected_val) > TOLERANCE:
                failures.append(f"{name}.{metric}: got {actual}, expected {expected_val}")

        print(f"{name}: OK")

    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("All models reload and match recorded metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
