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
