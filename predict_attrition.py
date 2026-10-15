
from pathlib import Path

import joblib
import pandas as pd


# 1. Define file paths
MODEL_PATH = Path("models/attrition_model.joblib")
DATA_PATH = Path("data/project1.csv")
THRESHOLD_PATH = Path("reports/selected_threshold.txt")

# 2. Check that required files exist
for file_path in [MODEL_PATH, DATA_PATH, THRESHOLD_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

# 3. Load the trained model and selected threshold
model = joblib.load(MODEL_PATH)
THRESHOLD = float(THRESHOLD_PATH.read_text().strip())

if not 0 <= THRESHOLD <= 1:
    raise ValueError("Classification threshold must be between 0 and 1.")

# 4. Load the dataset
df = pd.read_csv(DATA_PATH)

if "Attrition" not in df.columns:
    raise ValueError("Dataset is missing the Attrition column.")

if df.empty:
    raise ValueError("Dataset is empty.")

# 5. Ask which employee to predict
employee_index = int(
    input(f"Enter employee row number (1-{len(df)}): ")
)

if not 1 <= employee_index <= len(df):
    print("Invalid row number.")
    raise SystemExit(1)

# 6. Select the employee's features
# Use the same excluded columns as the training script.
employee = df.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ],
    errors="ignore",
).iloc[[employee_index - 1]]

# 7. Predict attrition probability
probability = model.predict_proba(employee)[0, 1]
prediction = "Yes" if probability >= THRESHOLD else "No"

# 8. Display results
print("\n--- EMPLOYEE ATTRITION PREDICTION ---")
print(f"Employee row: {employee_index}")
print(f"Predicted attrition probability: {probability:.1%}")
print(f"Classification threshold: {THRESHOLD:.2f}")
print(f"Predicted attrition: {prediction}")
print("\nNote: This is a model estimate, not a certainty about an employee.")
