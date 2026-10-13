
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find and load the dataset
csv_files = list(Path("data").glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV found inside the data folder.")

df = pd.read_csv(csv_files[0])

# Create a folder for chart outputs
output_folder = Path("reports/figures")
output_folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# Chart 1: Attrition distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(output_folder / "attrition_distribution.png", dpi=150)
plt.close()

# Chart 2: Monthly income by attrition
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
plt.title("Monthly Income by Attrition Status")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.savefig(output_folder / "income_by_attrition.png", dpi=150)
plt.close()

print("EDA completed successfully.")
print(f"Charts saved in: {output_folder}")
print("\nAverage monthly income by attrition:")
print(df.groupby("Attrition")["MonthlyIncome"].mean().round(2))
