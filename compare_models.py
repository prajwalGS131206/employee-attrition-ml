
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load dataset
csv_files = list(Path("data").glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV found in data/")

df = pd.read_csv(csv_files[0])

if "Attrition" not in df.columns:
    raise ValueError("Attrition column is missing.")

# Remove target, employee ID, and constant columns
X = df.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ],
    errors="ignore",
)

y = df["Attrition"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Attrition column.")

print(f"Dataset rows: {len(df)}")
print(f"Features used: {X.shape[1]}")


# 2. Use the same split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Identify feature types from training data
numeric_cols = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    exclude="number"
).columns.tolist()


# 4. Build a complete preprocessing and modeling pipeline
def make_pipeline(classifier):
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", classifier),
    ])


# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    ),
}

# 6. Train and evaluate both models
results = []

for name, classifier in models.items():
    pipeline = make_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

results_df = pd.DataFrame(results).set_index("Model").round(3)

print("\n--- BASELINE MODEL COMPARISON ---")
print(results_df.to_string())

# 7. Save results
Path("reports").mkdir(exist_ok=True)

results_df.to_csv("reports/model_comparison.csv")

print("\nSaved comparison to reports/model_comparison.csv")
print("Note: This is a separate baseline split, not the final holdout evaluation.")
