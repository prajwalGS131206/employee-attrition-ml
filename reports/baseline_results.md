
# Employee Attrition — Model Evaluation

## 1. Dataset
- Total records: 1,470
- Original columns: 35
- Employees who left: 16.12%
- Target: Attrition (No = 0, Yes = 1)

The target is imbalanced, so accuracy is not sufficient on its own.

## 2. Baseline Model Comparison

The following models were evaluated using the same baseline train/test split.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.861 | 0.615 | 0.340 | 0.438 | 0.812 |
| Random Forest | 0.827 | 0.452 | 0.404 | 0.427 | 0.791 |

Logistic Regression had higher accuracy, precision, F1-score, and ROC-AUC in this comparison. Random Forest had higher recall.

These are baseline comparison results, not the final holdout evaluation.

## 3. Threshold Selection

Candidate classification thresholds were evaluated on validation data. The threshold with the highest validation F1-score was selected.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.20 | 0.420 | 0.763 | 0.542 |
| 0.30 | 0.519 | 0.711 | 0.600 |
| 0.40 | 0.675 | 0.711 | 0.692 |
| 0.50 | 0.885 | 0.605 | 0.719 |
| 0.60 | 0.950 | 0.500 | 0.655 |

Selected threshold: **0.50**

The threshold was selected using validation data, not the final holdout set.

## 4. Final Holdout Evaluation

The final evaluation used a holdout set of 294 records.

| Metric | Score |
|---|---:|
| Threshold | 0.50 |
| Accuracy | 85.37% |
| Precision | 56.67% |
| Recall | 36.17% |
| F1-score | 44.16% |
| ROC-AUC | 82.01% |

### Confusion Matrix

Rows represent actual outcomes; columns represent predicted outcomes.

| | Predicted Stayed | Predicted Left |
|---|---:|---:|
| Actual Stayed | 234 | 13 |
| Actual Left | 30 | 17 |

The model detected 17 of the 47 employees who left in the holdout set and missed 30.

## 5. Interpretation and Limitations
- Accuracy should be interpreted alongside recall, precision, F1-score, and ROC-AUC because the target is imbalanced.
- The model's recall for attrition is limited and requires improvement.
- The results are estimates from a single holdout split, not independent external validation.
- Threshold choice depends on the relative costs of missed attrition cases and false alarms.
- Predictions are not proof that an employee will leave and should not be used as the sole basis for employment decisions.
