
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load dataset
data_files = list(Path("data").glob("*.csv"))

if not data_files:
    raise FileNotFoundError("Place the attrition CSV inside data/.")

df = pd.read_csv(data_files[0])

if "Attrition" not in df.columns:
    raise ValueError("Attrition column is missing.")

X = df.drop(
    columns=["Attrition", "EmployeeNumber"],
    errors="ignore",
)
y = df["Attrition"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Attrition column.")


# 2. Create a new holdout split
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2026,
    stratify=y,
)

# Split development data into fitting and validation portions
X_fit, X_val, y_fit, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    random_state=42,
    stratify=y_dev,
)


# 3. Build preprocessing pipelines
numeric_columns = X_fit.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X_fit.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])


def create_model():
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])


# 4. Train using fitting data only
model = create_model()
model.fit(X_fit, y_fit)


# 5. Select threshold using validation data
validation_probabilities = model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_val, predictions, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nVALIDATION RESULTS")
print(threshold_df.round(3).to_string(index=False))

best_index = threshold_df["F1-score"].idxmax()
best_threshold = float(threshold_df.loc[best_index, "Threshold"])

print(f"\nSelected validation threshold: {best_threshold}")


# 6. Evaluate once on the new holdout
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

final_results = pd.DataFrame([{
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, test_probabilities),
}])

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, test_predictions))

print("\nHOLDOUT EVALUATION RESULTS")
print(final_results.round(3).to_string(index=False))


# 7. Save holdout results
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

final_results.to_csv(
    "reports/final_evaluation.csv",
    index=False,
)

threshold_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False,
)


# 8. Train the final prediction model on all development data
final_model = create_model()
final_model.fit(X_dev, y_dev)

joblib.dump(
    final_model,
    "models/attrition_model.joblib",
)

Path("reports/selected_threshold.txt").write_text(
    f"{best_threshold}\n"
)

print("\nSaved final model trained on development data.")
print("Saved selected threshold.")
print("Saved holdout evaluation and threshold comparison.")
