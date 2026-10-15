
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# 1. Load the dataset
csv_files = list(Path("data").glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV found in the data folder.")

df = pd.read_csv(csv_files[0])

# 2. Separate features and target
X = df.drop(columns=["Attrition", "EmployeeNumber"], errors="ignore")
y = df["Attrition"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Attrition column.")

# 3. Re-create the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 4. Split the training portion into training and validation
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

# 5. Train a fresh model using only the fitting data
model = joblib.load("models/attrition_model.joblib")
model.fit(X_fit, y_fit)


# 6. Select threshold using validation data
val_probabilities = model.predict_proba(X_val)[:, 1]

results = []

for threshold in [0.3, 0.4, 0.5, 0.6]:
    val_predictions = (val_probabilities >= threshold).astype(int)

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
print("VALIDATION RESULTS")
print(results_df.round(3).to_string(index=False))

# 7. Select the threshold with the highest validation F1-score
best_row = results_df.loc[results_df["F1-score"].idxmax()]
best_threshold = float(best_row["Threshold"])

print(f"\nSelected threshold: {best_threshold}")

# 8. Evaluate once on the untouched test set
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

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

print("\nFINAL TEST RESULTS")
print(final_results.round(3).to_string(index=False))

# 9. Save the final evaluation
Path("reports").mkdir(exist_ok=True)
final_results.to_csv("reports/final_evaluation.csv", index=False)

print("\nSaved results to reports/final_evaluation.csv")
