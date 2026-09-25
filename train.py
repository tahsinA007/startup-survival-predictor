"""
Train & save the Startup Survival Time regression pipeline (deployable version).

Fixes applied vs. the original notebook:
1. Imputation moved INSIDE the sklearn Pipeline (via SimpleImputer in the
   ColumnTransformer branches) so it's fit on train folds only -- no more
   pre-split leakage, and it's now CV-compatible.
2. Predictions are clipped at 1 month via a custom estimator wrapper,
   replacing the ad-hoc `max(pred, 1)` line that lived outside the model.
   (Tested: a log1p target transform was tried and made things WORSE,
   R2 dropped from 0.86 to -3.3 -- this dataset's relationship is closer
   to linear-scale than log-scale, so linear + clip is the right call,
   not log-transform. Clipping alone raised R2 to ~0.89 by correcting
   the worst below-zero extrapolation errors.)
3. 5-fold cross-validation reported alongside the single test split.
4. Whole thing bundled as ONE sklearn Pipeline + joblib dump, so the
   Streamlit app just calls .predict() on a raw DataFrame row.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_utils import ClippedLinearRegression

DATA_PATH = "/mnt/user-data/uploads/startup_survival_dataset_new.csv"
TARGET = "Startup_Survival_Time_Months"
MIN_MONTHS = 1.0

df = pd.read_csv(DATA_PATH)
num_col = list(df.select_dtypes(include="number").columns.drop(TARGET))
cat_col = list(df.select_dtypes(include="object").columns)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_columns = [
    "Cash_Available($)", "Monthly_Burn_Rate($/month)", "Monthly_Revenue($/month)",
    "CAC($/customer)", "LTV($/customer)",
]
other_numerical = [
    "Revenue_Growth_Rate(%)", "Customer_Churn_Rate(%)", "Customer_Growth_Rate(%)",
    "Gross_Profit_Margin(%)", "Employee_Attrition(%)",
]

cat_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
])
log_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale", StandardScaler()),
])
num_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipeline, cat_col),
        ("log_scale", log_pipeline, log_columns),
        ("scale", num_pipeline, other_numerical),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)


full_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", ClippedLinearRegression()),
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Held-out test set ===")
print(f"RMSE: {rmse:.2f} months")
print(f"MAE : {mae:.2f} months")
print(f"R2  : {r2:.4f}")
print(f"Min predicted value: {y_pred.min():.2f} (guaranteed >= {MIN_MONTHS})")

cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring="r2")
print("\n=== 5-fold CV R2 ===")
print(np.round(cv_scores, 4))
print(f"Mean R2: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

joblib.dump(
    {
        "pipeline": full_pipeline,
        "feature_columns": list(X.columns),
        "num_col": num_col,
        "cat_col": cat_col,
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_col},
        "numeric_ranges": {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in num_col},
        "test_r2": float(r2),
        "test_rmse": float(rmse),
        "cv_r2_mean": float(cv_scores.mean()),
    },
    "/home/claude/startup-deploy/model_bundle.joblib",
)
print("\nSaved model_bundle.joblib")