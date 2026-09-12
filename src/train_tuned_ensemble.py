import time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import optuna
from scipy.optimize import minimize

optuna.logging.set_verbosity(optuna.logging.WARNING)
t0 = time.time()
RANDOM_STATE = 42

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["id"].copy()

RANGE_ANXIETY_MAP = {"Low": 0, "Medium": 1, "High": 2}
YESNO_MAP = {"No": 0, "Yes": 1}

def engineer(df):
    df = df.copy()
    df["Home_Charging_Possible"] = df["Home_Charging_Possible"].map(YESNO_MAP)
    df["Subsidy_Available"] = df["Subsidy_Available"].map(YESNO_MAP)
    df["Range_Anxiety_Level"] = df["Range_Anxiety_Level"].map(RANGE_ANXIETY_MAP)
    df["Total_Charging_Access"] = df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    df["Income_per_Commute_km"] = df["Annual_Income_USD"] / (df["Daily_Commute_km"] + 1)
    df["Cars_per_Income_100k"] = df["Number_of_Cars_Owned"] / (df["Annual_Income_USD"] / 1e5 + 0.01)
    df["Commute_x_RangeAnxiety"] = df["Daily_Commute_km"] * df["Range_Anxiety_Level"]
    df["EcoScore"] = df["Environmental_Concern_Level"] - df["Range_Anxiety_Level"]
    df["Charging_Readiness"] = df["Home_Charging_Possible"] + (df["Total_Charging_Access"] > 5).astype(int)
    df["High_Income"] = (df["Annual_Income_USD"] > df["Annual_Income_USD"].median()).astype(int)
    return df
    
train_fe = engineer(train)
test_fe = engineer(test)

cat_cols = ["Gender", "City_Type", "Current_Car_Type"]
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_fe[c], test_fe[c]], axis=0))
    train_fe[c] = le.transform(train_fe[c])
    test_fe[c] = le.transform(test_fe[c])

train_fe["target"] = train_fe["Will_Buy_EV"].map(YESNO_MAP)
feature_cols = [c for c in train_fe.columns if c not in ("id", "Will_Buy_EV", "target")]

X = train_fe[feature_cols].reset_index(drop=True)
y = train_fe["target"].reset_index(drop=True)
X_test = test_fe[feature_cols]

print(f"Full train: {X.shape}, test: {X_test.shape}", flush=True)
