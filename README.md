# EV Purchase Prediction

Binary classification: predict probability that a person will buy an electric
vehicle (`Will_Buy_EV`). Kaggle-style competition, evaluated on **ROC-AUC**.

- Train: 668,665 rows | Test: 286,571 rows | 14 raw features, no missing values
- Target imbalance: 82.5% No / 17.5% Yes

## Data

Not included in this zip (large, and specific to the competition). Expected at
`train.csv` / `test.csv` in the working directory, with columns:

```
id, Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
Charging_Stations_Near_Home, Charging_Stations_Near_Work,
Environmental_Concern_Level, Gender, City_Type, Current_Car_Type,
Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level, Will_Buy_EV
```
