
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


### Tech stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Git and GitHub



## Dataset
We plan to use the IBM HR Analytics Employee Attrition dataset.

Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset will be used for educational and portfolio purposes. Dataset attribution and applicable terms will be documented.

## Project Status
- [x] Dataset exploration and EDA
- [x] Logistic Regression and Random Forest comparison
- [x] Validation-based threshold selection
- [x] Model evaluation and prediction script
- [ ] Power BI dashboard
- [ ] Final documentation and project review

## Planned Project Structure
- `data/` — dataset instructions
- `notebooks/` — exploratory analysis
- `src/` — reusable Python code
- `tests/` — automated tests
- `reports/` — figures and evaluation reports
- `app.py` — interactive demo, if implemented

## How to Run
## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/prajwalGS131206/employee-attrition-ml.git
cd employee-attrition-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the IBM HR Analytics Attrition CSV file inside the `data/` directory. The dataset is excluded from GitHub, so obtain it separately from the dataset source listed above.

### 4. Run exploratory data analysis

```bash
python inspect_data.py
python eda_analysis.py
```

### 5. Train the baseline model

```bash
python train_model.py
```

### 6. Evaluate the model and select a threshold

```bash
python evaluate_thresholds.py
```

### 7. Generate a prediction

```bash
python predict_attrition.py
```

Enter an employee row number when prompted.

## Author
Prajwal

## Model Performance

The project compares Logistic Regression and Random Forest models to predict employee attrition.

### Final Holdout Evaluation

The Logistic Regression model was evaluated using a threshold selected on validation data. A fresh holdout split was used for this evaluation.

| Metric | Score |
|---|---:|
| Accuracy | 85.37% |
| Precision | 56.67% |
| Recall | 36.17% |
| F1-score | 44.16% |
| ROC-AUC | 82.00% |

### Key Findings

- Logistic Regression achieved a holdout ROC-AUC of 0.82.
- The validation process selected a classification threshold of 0.5 based on F1-score.
- The model identified 17 of 47 employees who left in the holdout set.
- Recall remains a limitation, so further improvements are needed.

### Evaluation Limitation

The holdout uses a new split of the same public dataset, which has been explored during earlier experiments. These results are a fresh-split evaluation, not independent external validation. The model is for educational purposes and should not be used to make employment decisions.


### Limitations

This project uses a public educational dataset. Its predictions are not validated for real-world HR decisions and should not be used to make employment decisions.

### Project Outputs

### Project Outputs

- Exploratory Data Analysis (EDA) charts showing attrition distribution and income patterns.
- Baseline model comparison using Logistic Regression and Random Forest.
- Validation-based classification threshold selection to balance precision and recall.
- Model evaluation report containing Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Confusion matrix to examine correct predictions and missed attrition cases.
- Saved scikit-learn preprocessing and Logistic Regression pipeline.
- Employee-level attrition prediction script using the saved model and selected threshold.

### Current Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 85.4% |
| Precision | 55.3% |
| Recall | 44.7% |
| F1-score | 49.4% |
| ROC-AUC | 0.799 |

*Note: These results are provisional because the test dataset was examined during earlier experiments. The model is a learning project and should not be used for employment decisions without further validation.*


### Technologies Used

- Python
- Pandas — data handling and analysis
- NumPy — numerical computations
- Scikit-learn — preprocessing, machine learning, and evaluation
- Matplotlib — data visualization
- Seaborn — statistical visualization
- Joblib — model saving and loading
- Git and GitHub — version control and project hosting
