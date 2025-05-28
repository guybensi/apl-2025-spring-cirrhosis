# optuna_tuning.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load data
train = pd.read_csv("train.csv")
high_missing = ["Tryglicerides", "Cholesterol", "Copper", "SGOT", "Alk_Phos", "Spiders", "Hepatomegaly", "Drug", "Ascites"]
train = train.drop(columns=high_missing)

# Encode labels
y = LabelEncoder().fit_transform(train["Status"])
X = train.drop(columns=["Status", "id"])

# Preprocessing
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])

# Custom log loss scorer
def custom_logloss(estimator, X_val, y_val):
    y_pred_proba = estimator.predict_proba(X_val)
    return -log_loss(y_val, y_pred_proba)

# Wrapper for tuning
class ModelTuner:
    def __init__(self, model_name):
        self.model_name = model_name

    def objective(self, trial):
        if self.model_name == "xgb":
            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42
            )

        elif self.model_name == "lgb":
            model = LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                random_state=42
            )

        elif self.model_name == "gbr":
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                random_state=42
            )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=kf, scoring=custom_logloss)
        return scores.mean()

# Run tuning for all models
for model_name in ["xgb", "lgb", "gbr"]:
    print(f"\n🔍 Optimizing {model_name.upper()}")
    tuner = ModelTuner(model_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(tuner.objective, n_trials=20)
    print("Best Log Loss:", -study.best_value)
    print("Best Params:", study.best_params)
    study.trials_dataframe().to_csv(f"optuna_{model_name}_results.csv", index=False)
