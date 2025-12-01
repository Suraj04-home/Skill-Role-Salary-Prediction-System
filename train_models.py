# train_models.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error

CSV_PATH = "india_job_dataset_50000.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(CSV_PATH)

# Basic cleaning - ensure necessary columns exist
required = {"skills", "job_title", "salary"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain {required} columns. Found: {df.columns.tolist()}")

df["skills"] = df["skills"].fillna("").astype(str)
df["job_title"] = df["job_title"].fillna("Unknown").astype(str)
df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)

# 2. Prepare target encoders
le_role = LabelEncoder()
y_role = le_role.fit_transform(df["job_title"])

# Save label encoder (so you can map preds back later)
joblib.dump(le_role, os.path.join(OUT_DIR, "label_encoder_role.joblib"))

# 3. Train/test split
X_train, X_test, y_role_train, y_role_test, y_salary_train, y_salary_test = train_test_split(
    df["skills"], y_role, df["salary"], test_size=0.15, random_state=42
)

# 4. Role classifier pipeline (TF-IDF + Logistic Regression)
role_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=25000)),
    ("clf", LogisticRegression(max_iter=400))
])
print("Training role classifier...")
role_pipe.fit(X_train, y_role_train)

# 5. Salary regressor pipeline (TF-IDF + RandomForest)
salary_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=25000)),
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
print("Training salary regressor...")
salary_pipe.fit(X_train, y_salary_train)

# 6. Evaluate quickly
y_role_pred = role_pipe.predict(X_test)
y_salary_pred = salary_pipe.predict(X_test)

role_acc = accuracy_score(y_role_test, y_role_pred)

# compute RMSE in a sklearn-version-agnostic way
mse = mean_squared_error(y_salary_test, y_salary_pred)
rmse = np.sqrt(mse)

print("Role accuracy (test):", role_acc)
print("Salary RMSE (test):", rmse)

# 7. Save models
joblib.dump(role_pipe, os.path.join(OUT_DIR, "role_pipeline.joblib"))
joblib.dump(salary_pipe, os.path.join(OUT_DIR, "salary_pipeline.joblib"))

print(f"Saved models to {OUT_DIR}/")
print("Files:", os.listdir(OUT_DIR))
