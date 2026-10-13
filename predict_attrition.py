import pandas as pd
import joblib

MODEL_PATH = "models/attrition_model.joblib"
THRESHOLD = 0.4

model = joblib.load(MODEL_PATH)

# Select an employee record for a test
df = pd.read_csv("data/project1.csv")

employee_index = int(input("Enter employee row number (1-1470): "))

if not 1 <= employee_index <= len(df):
    print("Invalid row number.")
    raise SystemExit

employee = df.drop(
    columns=["Attrition", "EmployeeNumber"]
).iloc[[employee_index - 1]]

probability = model.predict_proba(employee)[0, 1]
prediction = "Yes" if probability >= THRESHOLD else "No"

print(f"\nPredicted attrition probability: {probability:.1%}")
print(f"Classification threshold: {THRESHOLD}")
print(f"Predicted attrition: {prediction}")