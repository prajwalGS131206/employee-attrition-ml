
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
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
    raise ValueError("The dataset must contain an 'Attrition' column.")

# 2. Separate features (X) and target (y)
X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in the Attrition column.")

# Remove constant identifier columns
X = X.drop(columns=["EmployeeNumber"], errors="ignore")

# 3. Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

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

# 4. Build the full modeling pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

# 5. Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Train the baseline model
model.fit(X_train, y_train)

# 7. Evaluate on the held-out test set
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("MODEL: Logistic Regression")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Stayed", "Left"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 8. Save the fitted model
output_folder = Path("models")
output_folder.mkdir(exist_ok=True)

import joblib
joblib.dump(model, output_folder / "attrition_model.joblib")

print("\nModel saved to models/attrition_model.joblib")
