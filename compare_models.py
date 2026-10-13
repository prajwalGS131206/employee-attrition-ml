
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

# Load dataset
csv_files = list(Path("data").glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV found in data/")

df = pd.read_csv(csv_files[0])
X = df.drop(columns=["Attrition", "EmployeeNumber"], errors="ignore")
y = df["Attrition"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Attrition column.")

# Split once so both models use the same test records
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

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
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    ),
}

results = []

for name, model in models.items():
    pipeline = make_pipeline(model)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

results_df = pd.DataFrame(results).set_index("Model").round(3)

print("\n--- MODEL COMPARISON ---")
print(results_df)

Path("reports").mkdir(exist_ok=True)
results_df.to_csv("reports/model_comparison.csv")
print("\nSaved comparison to reports/model_comparison.csv")
