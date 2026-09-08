# EV Purchase Prediction

Binary classification: predict probability that a person will buy an electric
vehicle (`Will_Buy_EV`). Kaggle-style competition, evaluated on **ROC-AUC**.

- Train: 668,665 rows | Test: 286,571 rows | 14 raw features, no missing values
- Target imbalance: 82.5% No / 17.5% Yes

## Data

Expected at
`train.csv` / `test.csv` in the working directory, with columns:

```
id, Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
Charging_Stations_Near_Home, Charging_Stations_Near_Work,
Environmental_Concern_Level, Gender, City_Type, Current_Car_Type,
Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level, Will_Buy_EV
```
## Approach

### Feature engineering
- Ordinal-encoded `Range_Anxiety_Level` (Low/Medium/High → 0/1/2)
- Binary-encoded Yes/No flags (`Home_Charging_Possible`, `Subsidy_Available`)
- Label-encoded low-cardinality categoricals (`Gender`, `City_Type`, `Current_Car_Type`)
- 7 engineered features motivated by domain reasoning, not just raw passthrough:
  `Total_Charging_Access`, `Income_per_Commute_km`, `Cars_per_Income_100k`,
  `Commute_x_RangeAnxiety`, `EcoScore` (environmental concern − range anxiety),
  `Charging_Readiness`, `High_Income`

  ### Model selection
Compared Logistic Regression, HistGradientBoosting, LightGBM, and XGBoost via
3-fold CV on a subsample. All landed within ~0.002 AUC of each other
(0.938–0.940) — the signal is close to additive/linear, so tree ensembles only
win by a small margin over plain logistic regression.

### v1 — Baseline blend (`src/train_baseline_blend.py`)
50/50 blend of HistGradientBoosting + XGBoost, default-ish hyperparameters,
trained on full data, validated on a single held-out 10% split.
**Val AUC: 0.9422**

### v2 — Tuned, 5-fold bagged ensemble (`src/train_tuned_ensemble.py`)
- Optuna hyperparameter search (20 trials each for XGBoost/LightGBM, 15 for HGB)
  on a 150k-row subsample with 3-fold CV
- Retrained all three models with tuned params using 5-fold stratified CV on
  the **full** training set — out-of-fold (OOF) predictions used for honest
  validation, test predictions averaged (bagged) across the 5 folds to reduce
  variance
- Blend weights optimized on OOF AUC (Nelder-Mead): landed at ~45% XGBoost /
  55% LightGBM / ~0% HGB — HGB got weighted out of the final blend entirely

**OOF AUC: 0.9420** (XGB 0.9419, LGB 0.9419, HGB 0.9415 individually)

## Results summary

| Model | AUC |
|---|---|
| Logistic Regression (baseline) | 0.9383 |
| HistGradientBoosting (default) | 0.9400 |
| LightGBM (default) | 0.9390 |
| XGBoost (default) | 0.9398 |
| **v1: HGB+XGB blend, single split** | **0.9422** |
| v2: XGBoost (tuned, 5-fold OOF) | 0.9419 |
| v2: LightGBM (tuned, 5-fold OOF) | 0.9419 |
| v2: HGB (tuned, 5-fold OOF) | 0.9415 |
| **v2: tuned blend, 5-fold OOF** | **0.9420** |

**Takeaway:** tuning and 5-fold bagging did *not* meaningfully beat the
untuned single-split baseline (0.9420 vs 0.9422, within fold-to-fold noise).
Top feature importances (`Subsidy_Available`, `EcoScore`,
`Environmental_Concern_Level`) suggest the ceiling here is set by the
available features, not model capacity — v2 is still the more defensible
model (proper OOF validation, tuned hyperparameters, variance-reduced test
predictions via bagging) even though the headline AUC is flat.

## Files

```
notebooks/
  EV_Purchase_Prediction.ipynb   # full walkthrough: EDA -> feature eng ->
                                  # model selection -> tuning -> ensemble ->
                                  # submission, with markdown commentary
src/
  feature_engineering.py         # shared encoding + derived-feature logic
  train_baseline_blend.py        # v1: single-split HGB+XGB blend (standalone script)
  train_tuned_ensemble.py        # v2: Optuna tuning + 5-fold bagged ensemble (standalone script)
outputs/
  submission_v1_baseline.csv
  submission_v2_tuned_ensemble.csv
  val_results.json               # v1 validation AUCs + blend weight search
  tuning_results.json            # Optuna best params/scores per model
  final_results.json             # v2 OOF AUCs + optimized blend weights
  feature_importance.json        # XGBoost gain-based feature importance
requirements.txt
```

`src/train_baseline_blend.py` and `src/train_tuned_ensemble.py` are
self-contained (each includes its own copy of the feature-engineering logic)
so they can be run standalone with `python src/<file>.py`.
`src/feature_engineering.py` is the same logic factored out as an importable
module, and is what the notebook uses.


