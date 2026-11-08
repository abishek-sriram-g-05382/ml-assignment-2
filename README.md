# Frog Species Classification (Anuran Calls MFCCs)


## Problem statement

The main problem statement is a interesting topic where a frog makes a call, is it possible to identify the frog species based on the call? from the audio fingerprint? 

So, The choosen dataset is containing audio fingerprint of the frog calls and each row in the dataset is one syllable of a recording, with 22 features MFCCs (Mel-frequency cepstral coefficients) - a standard way of summarizing frequency content. Based on success of this solution, this can be used for monitoring and surveying frog species and used for their population analysis. This will help automate and eliminate need of human listening manually.

## Dataset

**Anuran Calls (MFCCs)**, UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip 

| Property | Value |  
| -------- | -------- |  
| Rows  | 7195  |  
| Features  | 22  |  
| Target | Species | 
|Dropped| Family, Genus, RecordId |

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

**Feature table:** `MFCCs_1` through `MFCCs_22`, all continuous [-1,1]

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

| ML Model Name	| Observation about model performance
| -------------| -----------|
|Logistic Regression	|Accuracy is 96.6%, but macro F1 is only 0.909, so accuracy hides weakness on rare species like Rhinellagranulosa (68 rows). As a linear model, it cannot draw curved boundaries between species.|
|Decision Tree	|Accuracy is 93.95%, macro F1 is 0.864 and AUC is 0.932, with the best depth at max_depth=10. One tree makes only straight, axis-aligned cuts. Deeper trees overfit and shallower ones underfit.|
|KNN	|Very strong: accuracy 98.68%, macro F1 0.973 and MCC 0.982, with k=3 the best. MFCC features are compact and similar in scale, so “nearest call sounds the same” works well. Prediction is slower on large data because it scans all training rows.|
|Naive Bayes |	Weakest model: accuracy 91.7% and macro F1 0.836. It assumes the 22 MFCCs are independent, but they are correlated by design. This hurts most on the small species.
|Support Vector Machine	|Best overall: accuracy 99.10%, macro F1 0.978, MCC 0.987 and AUC 0.9996 (RBF kernel, C=10). The RBF kernel bends the boundary more flexibly than KNN. It was the slowest to train, since probability=True adds extra calibration.|
|Overall, Winner for your dataset? |	SVM. Classes are very imbalanced (about 50:1), so accuracy alone is misleading. SVM leads on MCC (0.987), macro F1 (0.978) and AUC (0.9996). KNN is a close second (MCC 0.982) and is a good choice if cheap training matters more.|


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


## Project Structure

```
anuran-species-classifier/
├── app.py                      # Streamlit app — upload, predict, compare, visualize
├── requirements.txt
├── test_data.csv                # held-out test split (raw features + target)
├── data/
│   ├── Frogs_MFCCs.csv          # raw dataset from UCI
│   └── SOURCE.md                # where it came from, and what to watch out for
├── notebooks/
│   └── eda.ipynb                # class balance, missing values, feature ranges
├── src/
│   ├── data.py                  # load, clean columns, drop leak columns, split
│   ├── metrics.py               # shared metric computation (used by train.py and app.py)
│   └── train.py                 # builds, tunes, evaluates, and saves all 5 models
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── results/
│   └── metrics_comparison.csv   # output of train.py — source for the Results table above
├── scripts/
│   └── smoke_test.py            # reloads each model and checks metrics still match
```

## Test Data

[`test_data.csv`](test_data.csv) —  20% split (1,439 rows)  test data
