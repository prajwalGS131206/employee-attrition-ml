
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/attrition_model.joblib")
DATA_PATH = Path("data/project1.csv")
THRESHOLD_PATH = Path("reports/selected_threshold.txt")

EXCLUDED_COLUMNS = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]


def load_model_and_threshold():
    """Load the saved model and validate its classification threshold."""
    for file_path in [MODEL_PATH, THRESHOLD_PATH]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    model = joblib.load(MODEL_PATH)
    threshold = float(THRESHOLD_PATH.read_text().strip())

    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1.")

    return model, threshold


def predict_employee(model, employee, threshold):
    """Return the predicted attrition probability and class."""
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1.")

    probabilities = model.predict_proba(employee)

    if probabilities.shape != (1, 2):
        raise ValueError("Expected one employee and two class probabilities.")

    probability = float(probabilities[0, 1])

    if not 0 <= probability <= 1:
        raise ValueError("Predicted probability must be between 0 and 1.")

    prediction = "Yes" if probability >= threshold else "No"
    return probability, prediction


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {DATA_PATH}")

    model, threshold = load_model_and_threshold()
    df = pd.read_csv(DATA_PATH)

    if "Attrition" not in df.columns:
        raise ValueError("Dataset is missing the Attrition column.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    employee_index = int(input(f"Enter employee row number (1-{len(df)}): "))

    if not 1 <= employee_index <= len(df):
        print("Invalid row number.")
        raise SystemExit(1)

    employee = df.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    ).iloc[[employee_index - 1]]

    probability, prediction = predict_employee(
        model, employee, threshold
    )

    print("\n--- EMPLOYEE ATTRITION PREDICTION ---")
    print(f"Employee row: {employee_index}")
    print(f"Predicted attrition probability: {probability:.1%}")
    print(f"Classification threshold: {threshold:.2f}")
    print(f"Predicted attrition: {prediction}")
    print("\nNote: This is a model estimate, not a certainty about an employee.")


if __name__ == "__main__":
    main()
