import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/project1.csv")

# Separate input features and target
X = df.drop(columns=["Attrition", "EmployeeNumber"])
y = df["Attrition"].map({"No": 0, "Yes": 1})

# Use the same test split as our earlier models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load the trained model
model = joblib.load("models/attrition_model.joblib")

# Get predicted probabilities of attrition
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate different classification thresholds
results = []

for threshold in [0.3, 0.4, 0.5, 0.6]:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))