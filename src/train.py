"""Train the five models on the Anuran Calls dataset, evaluate each on a
held-out stratified split, and save everything needed for the Streamlit app:
the fitted pipelines, test_data.csv, and a metrics comparison table.

Run from the repo root: python -m src.train
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import REPO_ROOT, load_features_and_target, save_test_csv, split
from src.metrics import compute_metrics

MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = REPO_ROOT / "results"
TEST_CSV_PATH = REPO_ROOT / "test_data.csv"

RANDOM_STATE = 42


def build_logistic_regression() -> tuple[Pipeline, dict]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    return pipe, param_grid


def build_decision_tree() -> tuple[Pipeline, dict]:
    # no scaling needed for a tree — splits don't care about feature scale
    pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    # tuning depth/leaf size because an unpruned tree on this data will
    # memorize the training set (near-100% train accuracy) and that's not
    # a real result, it's overfitting
    param_grid = {
        "clf__max_depth": [5, 10, 15, 20, None],
        "clf__min_samples_leaf": [1, 5, 10],
    }
    return pipe, param_grid


def build_knn() -> tuple[Pipeline, dict]:
    # scaling matters a lot here — KNN is pure distance, so one feature
    # with a wider range than the others would otherwise dominate it
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])
    param_grid = {"clf__n_neighbors": [3, 5, 7, 9, 11, 15]}
    return pipe, param_grid


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "decision_tree": build_decision_tree,
    "knn": build_knn,
}


def fit_and_evaluate(name, builder, X_train, X_test, y_train, y_test):
    pipe, param_grid = builder()
    search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba, class_labels=best.classes_)
    metrics["best_params"] = search.best_params_

    print(f"{name}: best_params={search.best_params_}")
    print(f"  {metrics}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / f"{name}.joblib", compress=3)

    return metrics


def main():
    X, y, feature_cols = load_features_and_target()
    X_train, X_test, y_train, y_test = split(X, y, random_state=RANDOM_STATE)

    save_test_csv(X_test, y_test, TEST_CSV_PATH)
    print(f"wrote {TEST_CSV_PATH} ({len(X_test)} rows)")

    all_metrics = {}
    for name, builder in MODEL_BUILDERS.items():
        all_metrics[name] = fit_and_evaluate(
            name, builder, X_train, X_test, y_train, y_test
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(all_metrics).T
    results_df.to_csv(RESULTS_DIR / "metrics_comparison.csv")
    print(results_df)


if __name__ == "__main__":
    main()
