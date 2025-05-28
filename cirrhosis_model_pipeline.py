# cirrhosis_model_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
import optuna
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# === Load Data ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# === Drop high-missing columns ===
high_missing = ["Tryglicerides", "Cholesterol", "Copper", "SGOT", "Alk_Phos", "Spiders", "Hepatomegaly", "Drug", "Ascites"]
train = train.drop(columns=high_missing)
test = test.drop(columns=[col for col in high_missing if col in test.columns])

# === Encode labels ===
le = LabelEncoder()
train["Status"] = le.fit_transform(train["Status"])

# === Target ===
y = train["Status"]
X = train.drop(columns=["Status", "id"])
X_test = test.drop(columns=["id"])

# === Preprocessing ===
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

# === Model testing function ===
def evaluate_model(clf, name):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe.fit(X_train, y_train)
        y_pred_proba = pipe.predict_proba(X_val)
        scores.append(log_loss(y_val, y_pred_proba, labels=[0, 1, 2]))
    print(f"{name} Avg Log Loss: {np.mean(scores):.5f}")
    return np.mean(scores)

# === Try a few baseline models ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, multi_class='multinomial')
}

model_scores = {name: evaluate_model(model, name) for name, model in models.items()}

# === Save scores for weighting ===
pd.Series(model_scores).to_csv("model_logloss_scores.csv")
