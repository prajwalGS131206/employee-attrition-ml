import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = "data/project1.csv"
MODEL_PATH = "models/attrition_model.joblib"
THRESHOLD = 0.4

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Attrition", "EmployeeNumber"])
y = df["Attrition"].map({"No": 0, "Yes": 1})

# Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Load model and predict probabilities
model = joblib.load(MODEL_PATH)
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

# Calculate metrics
metrics = {
    "Threshold": THRESHOLD,
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1-score": f1_score(y_test, predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities),
}

print("\n=== MODEL EVALUATION ===")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, predictions))

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No attrition", "Attrition"],
        zero_division=0,
    )
)

# Save results
results = pd.DataFrame([metrics])
results.to_csv("reports/final_evaluation.csv", index=False)

print("\nResults saved to reports/final_evaluation.csv")