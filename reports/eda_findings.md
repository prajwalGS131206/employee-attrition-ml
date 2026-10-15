
# Employee Attrition — Exploratory Data Analysis

## 1. Objective
Explore employee HR data to identify patterns associated with employee attrition and support the development of a machine learning classification model.

## 2. Dataset Overview
- Dataset: IBM HR Analytics Employee Attrition
- Total records: 1,470
- Original columns: 35
- Missing values: 0
- Duplicate rows: 0
- Constant columns: EmployeeCount, Over18, StandardHours

The EmployeeNumber column is an identifier and is excluded from model features. The three constant columns are also excluded because they do not provide predictive variation.

## 3. Attrition Distribution
- Employees who stayed: 1,233 (83.88%)
- Employees who left: 237 (16.12%)

The target variable is imbalanced. Accuracy alone can therefore be misleading, so precision, recall, F1-score, ROC-AUC, and the confusion matrix are also used for model evaluation.

## 4. Monthly Income Analysis
- Average monthly income for employees who stayed: 6,832.74
- Average monthly income for employees who left: 4,787.09

In this dataset, employees who left had a lower average monthly income than employees who stayed. This is an observed association and does not establish that income causes attrition.

Further analysis could investigate income alongside job role, seniority, overtime, satisfaction, and other employee characteristics.

## 5. Visualizations
- `figures/attrition_distribution.png`
- `figures/income_by_attrition.png`

## 6. Data Quality and Preprocessing
- No missing values were found during the initial dataset inspection.
- No fully duplicated rows were found.
- Constant columns and the employee identifier are excluded from model features.
- Numeric features use median imputation and standardization.
- Categorical features use most-frequent imputation and one-hot encoding.
- Preprocessing is included in the machine learning pipeline to prevent fitting preprocessing transformations on the holdout data.

## 7. Initial Model Evaluation
A Logistic Regression model was evaluated on a holdout set of 294 records.

- Accuracy: 85.37%
- Precision for attrition: 56.67%
- Recall for attrition: 36.17%
- F1-score for attrition: 44.16%
- ROC-AUC: 82.01%
- Classification threshold: 0.50

The model detected 17 of the 47 employees who left in the holdout set and missed 30. Recall is a key area for improvement.

A separate baseline comparison found that Logistic Regression outperformed Random Forest on F1-score and ROC-AUC in that comparison split. These comparison results are not directly interchangeable with the final holdout results because the splits differ.

## 8. Limitations
- The findings describe associations, not causes.
- The dataset may not represent every organization's workforce.
- The evaluation is based on a single holdout split from a public dataset.
- Previous exploration of the dataset means the holdout should not be treated as completely independent external validation.
- Predictions are estimates and should not be used as the sole basis for employment decisions.

## 9. Future Improvements
1. Investigate ways to improve attrition recall without creating excessive false positives.
2. Validate the model on an independent dataset if available.
3. Examine feature importance and model explanations carefully.
4. Test the complete prediction workflow and document reproducible setup instructions.
