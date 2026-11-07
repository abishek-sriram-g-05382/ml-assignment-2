# Frog Species Classification from Audio (Anuran Calls MFCCs)

ML Assignment 2 — training and comparing five classifiers on the same dataset, then serving them through a Streamlit app that loads the saved models and scores whatever test CSV you hand it.

- GitHub repo: TODO
- Live app: TODO

## Problem statement

TODO — one paragraph: predicting frog species from MFCC features extracted from audio recordings of their calls, why that's a useful/interesting problem (bioacoustic monitoring, biodiversity surveys without needing a human to listen to every recording).

## Dataset

TODO — source link, row count, feature count, target, class distribution table, preprocessing applied, feature table.

## Results

TODO — comparison table (6 metrics x 5 models) and per-model observations + overall winner, pasted from `results/metrics_comparison.csv` once training is done.

## Limitations

TODO — note on group leakage from `RecordID` (syllables from the same recording land in both train/test under a plain stratified split).

## Running locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
