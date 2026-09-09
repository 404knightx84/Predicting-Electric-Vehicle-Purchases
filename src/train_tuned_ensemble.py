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
