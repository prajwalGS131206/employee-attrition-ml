
# Employee Attrition Prediction & HR Analytics

## Project Overview
An end-to-end data science project focused on understanding employee attrition and predicting which employees may be at risk of leaving an organization.

The project combines Python, machine learning, exploratory data analysis, and Power BI to turn HR data into actionable business insights.

## Objectives
- Explore and clean employee HR data.
- Identify patterns associated with employee attrition.
- Build and evaluate machine learning classification models.
- Interpret the factors associated with attrition.
- Develop an interactive HR analytics dashboard in Power BI.

## Tech Stack
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Power BI
- Git and GitHub

## Dataset
We plan to use the IBM HR Analytics Employee Attrition dataset.

Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset will be used for educational and portfolio purposes. Dataset attribution and applicable terms will be documented.

## Project Status
In progress — data exploration, model development, evaluation, and dashboard creation are planned.

## Planned Project Structure
- `data/` — dataset instructions
- `notebooks/` — exploratory analysis
- `src/` — reusable Python code
- `tests/` — automated tests
- `reports/` — figures and evaluation reports
- `app.py` — interactive demo, if implemented

## How to Run
Setup and execution instructions will be added after the implementation is tested.

## Author
Prajwal

## Model Performance

The project compares Logistic Regression and Random Forest models to predict employee attrition.

### Evaluation Results

The Logistic Regression model was evaluated using a classification threshold of 0.4.

| Metric | Score |
|---|---:|
| Accuracy | 86.73% |
| Precision | 61.76% |
| Recall | 44.68% |
| F1-score | 51.85% |
| ROC-AUC | 81.14% |

### Key Findings

- Logistic Regression achieved a ROC-AUC of approximately 0.81.
- Lowering the classification threshold from 0.5 to 0.4 improved recall and F1-score on the held-out test set.
- Attrition prediction remains challenging because the dataset contains fewer employees who leave than employees who stay.
- Recall remains a limitation, so further model improvements are needed.

### Limitations

This project uses a public educational dataset. Its predictions are not validated for real-world HR decisions and should not be used to make employment decisions.

### Project Outputs

- Exploratory data analysis charts
- Baseline model comparison
- Classification threshold experiment
- Model evaluation metrics saved as CSV
- Saved machine learning pipeline
