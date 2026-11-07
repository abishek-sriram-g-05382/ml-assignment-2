# Data source

Anuran Calls (MFCCs) dataset, UCI Machine Learning Repository.

- Download: https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip
- Downloaded: 2026-11-07
- File used: `Frogs_MFCCs.csv`

Each row is one syllable extracted from an audio recording of a frog call,
described by 22 MFCC (Mel-frequency cepstral coefficient) features. Checked on download:

- 7195 rows, 26 columns
- 22 numeric MFCC feature columns (`MFCCs_ 1` ... `MFCCs_22`, note the stray
  leading space in some header names — stripped on load in `src/data.py`)
- `Family`, `Genus`, `Species` — three label columns at increasing granularity.
  `Species` is the target; `Family` and `Genus` are dropped before training
  since they're near-deterministic functions of `Species` (leakage, not
  features).
- `RecordID` — identifies which original audio recording a syllable came
  from. Dropped as a feature (it's an identifier, not a signal), but kept in
  mind for the split: see the limitations note in the README.
- No missing values.
- 10 distinct species, heavily imbalanced: largest class (`AdenomeraHylaedactylus`)
  has 3478 rows, smallest (`Rhinellagranulosa`) has 68.
- Only 60 unique `RecordID` values across 7195 rows — syllables from the same
  recording are near-duplicates of each other. A plain random/stratified
  split will put near-identical syllables in both train and test, inflating
  reported scores. Not fixed here (see README limitations); a proper fix
  would use `GroupShuffleSplit`/`GroupKFold` keyed on `RecordID`.
