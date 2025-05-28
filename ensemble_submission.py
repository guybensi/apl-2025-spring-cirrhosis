# ensemble_submission_fixed.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

# === Load data ===
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# === Drop high-missing columns ===
high_missing = ["Tryglicerides", "Cholesterol", "Copper", "SGOT", "Alk_Phos", "Spiders", "Hepatomegaly", "Drug", "Ascites"]
train = train.drop(columns=high_missing)
test = test.drop(columns=[col for col in high_missing if col in test.columns])

# === Encode labels ===
le = LabelEncoder()
train["Status"] = le.fit_transform(train["Status"])

X = train.drop(columns=["Status", "id"])
y = train["Status"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

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

# === Define optimized models ===
xgb_model = XGBClassifier(
    n_estimators=890, max_depth=3, learning_rate=0.055, subsample=0.77,
    colsample_bytree=0.84, eval_metric='mlogloss', use_label_encoder=False, random_state=42
)

lgb_model = LGBMClassifier(
    n_estimators=910, max_depth=3, learning_rate=0.045, subsample=0.85,
    colsample_bytree=0.88, random_state=42
)

gbr_model = GradientBoostingClassifier(
    n_estimators=964, max_depth=3, learning_rate=0.02347, subsample=0.8379, random_state=42
)

# === Get test preds only ===
def get_test_preds(model):
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X, y)
    return pipe.predict_proba(X_test)

xgb_test = get_test_preds(xgb_model)
lgb_test = get_test_preds(lgb_model)
gbr_test = get_test_preds(gbr_model)

# === Use best known weights ===
w_xgb = 0.45
w_lgb = 0.42
w_gbr = 1.0 - w_xgb - w_lgb

final_preds = w_xgb * xgb_test + w_lgb * lgb_test + w_gbr * gbr_test
final_labels = le.inverse_transform(np.argmax(final_preds, axis=1))

submission = pd.DataFrame({"id": test_ids, "Status": final_labels})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv generated using fixed weights:", f"XGB={w_xgb}", f"LGB={w_lgb}", f"GBR={w_gbr:.4f}")
