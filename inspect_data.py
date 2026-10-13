
from pathlib import Path

import pandas as pd

# Find CSV files inside the data folder
data_folder = Path("data")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found in the data folder. "
        "Please upload the employee dataset first."
    )

# Read the first CSV file
file_path = csv_files[0]
df = pd.read_csv(file_path)

print("\n--- DATASET OVERVIEW ---")
print(f"File: {file_path.name}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

if "Attrition" in df.columns:
    print("\n--- ATTRITION COUNTS ---")
    print(df["Attrition"].value_counts())

    print("\n--- ATTRITION PERCENTAGES ---")
    print(
        (df["Attrition"].value_counts(normalize=True) * 100)
        .round(2)
    )
else:
    print("\nWarning: No 'Attrition' column was found.")
