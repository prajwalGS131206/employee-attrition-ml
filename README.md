
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

### Evaluation Results

The Logistic Regression model was evaluated using a classification threshold of 0.4.
| Metric | Score |
|---|---:|
| Accuracy | 85.37% |
| Precision | 55.26% |
| Recall | 44.68% |
| F1-score | 49.41% |
| ROC-AUC | 79.94% |

### Key Findings

- Logistic Regression achieved a ROC-AUC of approximately 0.81.
-A threshold of 0.4 was selected using validation data to balance precision and recall.
- Attrition prediction remains challenging because the dataset contains fewer employees who leave than employees who stay.
- Recall remains a limitation, so further model improvements are needed.

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
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Git and GitHub
