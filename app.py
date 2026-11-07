"""Streamlit front-end for the frog species classifier.

Loads the five saved pipelines from model/ and scores whatever CSV is
uploaded — never retrains, never calls .fit() on uploaded data.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Gaussian Naive Bayes",
    "svm": "SVM",
}

st.set_page_config(page_title="Frog Species Classifier", page_icon="🐸", layout="wide")


@st.cache_resource
def load_models() -> dict:
    models = {}
    for name in MODEL_LABELS:
        path = MODEL_DIR / f"{name}.joblib"
        models[name] = joblib.load(path)
    return models


def main():
    st.title("🐸 Frog Species Classifier")
    st.caption("Predicting Anuran species from MFCC call features — upload a test CSV below.")

    models = load_models()

    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox(
        "Model", options=list(MODEL_LABELS), format_func=lambda k: MODEL_LABELS[k]
    )

    uploaded = st.file_uploader("Upload test CSV", type="csv")

    if uploaded is None:
        st.info("Upload a CSV with the 22 MFCC feature columns plus the Species column to get started.")
        return

    df = pd.read_csv(uploaded)
    st.write(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    st.dataframe(df.head())

    model = models[model_name]
    st.write(f"Selected model: **{MODEL_LABELS[model_name]}**")


if __name__ == "__main__":
    main()
