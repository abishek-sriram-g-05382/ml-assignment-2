# Frog Species Classification from Audio (Anuran Calls MFCCs)

ML Assignment 2 — training and comparing five classifiers on the same dataset, then serving them through a Streamlit app that loads the saved models and scores whatever test CSV you hand it.

- GitHub repo: TODO (fill in after pushing)
- Live app: TODO (fill in after Streamlit Cloud deploy)

## Problem statement

Can a model tell which of ten frog species made a call, just from the audio fingerprint of that call? Each row in this dataset is one syllable from a field recording, reduced to 22 MFCCs (Mel-frequency cepstral coefficients) — a standard way of summarizing the frequency content of a short audio clip. If this works reasonably well, it's the kind of thing that matters for actual bioacoustic monitoring: surveying frog populations (an indicator species for wetland health) from passive audio recorders instead of needing a person to listen to, and manually label, thousands of hours of recordings.

## Dataset

**Anuran Calls (MFCCs)**, UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip (full details in [`data/SOURCE.md`](data/SOURCE.md)).

- 7,195 rows, 22 numeric features (`MFCCs_1` … `MFCCs_22`), no missing values
- Target: `Species` (10 classes)
- `Family` and `Genus` are dropped — they're coarser labels derived from `Species`, i.e. leakage, not features
- `RecordID` is dropped as a feature (it's an identifier) — see Limitations below for why it still matters

Class distribution (full dataset):

| Species | Rows |
|---|---|
| AdenomeraHylaedactylus | 3478 |
| HypsiboasCordobae | 1121 |
| AdenomeraAndre | 672 |
| Ameeregatrivittata | 542 |
| HypsiboasCinerascens | 472 |
| HylaMinuta | 310 |
| LeptodactylusFuscus | 270 |
| ScinaxRuber | 148 |
| OsteocephalusOophagus | 114 |
| Rhinellagranulosa | 68 |

That's about a 50:1 imbalance between the largest and smallest class, which is why accuracy alone isn't trusted anywhere in this writeup — macro-averaged metrics and MCC matter more. See [`notebooks/eda.ipynb`](notebooks/eda.ipynb) for the full exploration (class balance chart, feature ranges, the group-leakage check).

**Preprocessing:** stratified 80/20 train/test split (`random_state=42`), no imputation needed (no missing values). Each model is a scikit-learn `Pipeline`; `StandardScaler` precedes Logistic Regression, KNN, and SVM (all distance/gradient sensitive), while the Decision Tree and Gaussian Naive Bayes skip scaling (trees don't care about scale, and GaussianNB's variance estimate per feature makes scaling irrelevant to it).

**Feature table:** `MFCCs_1` through `MFCCs_22`, all continuous, roughly in the `[-1, 1]` range by construction (they're normalized cepstral coefficients, not raw audio).

## Results

Metrics computed on the held-out test split (`test_data.csv`, 1,439 rows), precision/recall/F1 macro-averaged across the 10 species, AUC via `roc_auc_score(..., multi_class="ovr", average="macro")`. Full numbers in [`results/metrics_comparison.csv`](results/metrics_comparison.csv).

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9659 | 0.9968 | 0.9132 | 0.9132 | 0.9085 | 0.9527 |
| Decision Tree | 0.9395 | 0.9315 | 0.8744 | 0.8562 | 0.8642 | 0.9157 |
| KNN | 0.9868 | 0.9899 | 0.9793 | 0.9672 | 0.9727 | 0.9816 |
| Gaussian Naive Bayes | 0.9173 | 0.9890 | 0.8470 | 0.8490 | 0.8361 | 0.8872 |
| SVM | **0.9910** | **0.9996** | **0.9787** | **0.9775** | **0.9780** | **0.9874** |

### Observations

**Logistic Regression** — 96.6% accuracy but macro-F1 of only 0.909, a real gap that accuracy alone hides. As a linear model it assumes the ten species separate along roughly linear boundaries in MFCC space, which is partly true here, but it's visibly weaker than the non-linear methods specifically on the rarer species (e.g. `Rhinellagranulosa`, 68 rows) where a straight-line boundary has less margin to work with.

**Decision Tree** — the weakest of the non-NB models: 93.95% accuracy, 0.864 macro-F1, 0.932 AUC, with the grid settling on `max_depth=10`. A single tree has to carve a 10-way split through 22-dimensional MFCC space with axis-aligned cuts, which is a cruder tool than a distance metric or a margin; deeper trees overfit (verified by comparing against the unpruned `max_depth=None` result during tuning) and shallower ones underfit, so this is close to the best a lone tree can do on this feature set.

**KNN** — very strong: 98.68% accuracy, 0.973 macro-F1, 0.982 MCC, with `k=3` beating every larger k in the grid. This is close to a best-case scenario for KNN — the MFCC feature space is compact, continuous, and already roughly uniform in scale, so "nearest call sounds the same" turns out to be a genuinely good rule. The cost is inference, not training: every prediction means scanning the stored training set, which is cheap at ~5,700 rows but wouldn't stay cheap at much larger scale.

**Gaussian Naive Bayes** — clearly the weakest model: 91.7% accuracy, macro-F1 only 0.836. GaussianNB assumes the 22 MFCCs are conditionally independent given the species, and that assumption is simply false here — MFCCs are coefficients of overlapping frequency bands computed from the same short audio frame, so they're correlated by construction. NB overestimates the joint likelihood as a result, and it shows up most on the smaller species, where there's less data to average the violated assumption away.

**SVM** — the overall best performer: 99.10% accuracy, 0.978 macro-F1, 0.987 MCC, 0.9996 AUC, leading on 5 of 6 metrics (the grid picked an RBF kernel with `C=10` over a linear kernel). The RBF kernel can bend its decision boundary more flexibly than KNN's raw distance measure, and that extra flexibility pays off slightly here. The real cost is training time: `probability=True` adds a Platt-scaling calibration step on top of the usual SVM fit, which was noticeably the slowest of the five models to train — at ~5,700 training rows it's a non-issue, but it's the one model here where training cost would need a second look on a much bigger dataset.

### Overall winner: SVM

Given the ~50:1 class imbalance, accuracy is the least trustworthy metric in this table — it rewards all five models for simply nailing `AdenomeraHylaedactylus` (3,478 of 7,195 rows). MCC and macro-F1 are the fairer comparison, and SVM leads on both (MCC 0.987, F1 0.978), plus leads outright on AUC (0.9996). It's close with KNN (MCC 0.982, F1 0.973) — close enough that **KNN is a completely reasonable second choice**, and a cheaper one at inference time if that mattered more than the last fraction of a point of accuracy. SVM wins this comparison, but not by a landslide.

## Limitations

**Group leakage via `RecordID`.** The 7,195 rows come from only 60 distinct recordings — syllables from the same recording are near-duplicates of each other (same frog, same call, same recording conditions). The stratified train/test split used here is a plain row-level split, so syllables from the same recording can land on both sides, which inflates every model's reported score by some amount. The correct fix is a group-aware split (`GroupShuffleSplit` for the train/test split, `GroupKFold` keyed on `RecordID` for the tuning grids), which wasn't implemented here given the assignment's time budget. All five models are affected equally by this, so the *relative* ranking in the Results section should still hold even if the *absolute* numbers are somewhat optimistic.

## Running locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain everything from scratch (writes `test_data.csv`, `model/*.joblib`, `results/metrics_comparison.csv`):

```
python -m src.train
```
