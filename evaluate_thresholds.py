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


# 1. Load the dataset
csv_files = list(Path("data").glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV found in the data folder.")

df = pd.read_csv(csv_files[0])

if "Attrition" not in df.columns:
    raise ValueError("Attrition column not found.")

# 2. Separate features and target
X = df.drop(
    columns=["Attrition", "EmployeeNumber"],
    errors="ignore",
)

y = df["Attrition"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Attrition column.")

# 3. Split off the test set
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Split development data into fitting and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.2,
    random_state=42,
    stratify=y_dev,
)

# 5. Identify numeric and categorical columns
numeric_columns = X_fit.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X_fit.select_dtypes(
    exclude="number"
).columns.tolist()

# 6. Create preprocessing pipelines
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

# 7. Build and train a fresh Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

model.fit(X_fit, y_fit)

# 8. Select threshold using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

results = []

for threshold in [0.3, 0.4, 0.5, 0.6]:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_val, val_predictions, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)

print("\nVALIDATION RESULTS")
print(results_df.round(3).to_string(index=False))

# Choose threshold with the highest validation F1-score
best_row = results_df.loc[
    results_df["F1-score"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print(f"\nSelected threshold: {best_threshold}")

# 9. Evaluate on test data
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
    "ROC-AUC": roc_auc_score(
        y_test, test_probabilities
    ),
}])

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, test_predictions))

print("\nFINAL TEST RESULTS")
print(final_results.round(3).to_string(index=False))

# 10. Save the evaluation report and selected threshold
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

final_results.to_csv(
    "reports/final_evaluation.csv",
    index=False,
)

Path("reports/selected_threshold.txt").write_text(
    f"{best_threshold}\n"
)

# 11. Save the fitted model
joblib.dump(
    model,
    "models/attrition_model.joblib",
)

print("\nSaved evaluation report.")
print("Saved selected threshold.")
print("Saved trained model.")