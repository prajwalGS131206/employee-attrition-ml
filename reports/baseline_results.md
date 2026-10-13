# Baseline Model Evaluation

## Dataset
- Rows: 1,470
- Features: 35
- Employees who left: 16.12%
- Target: Attrition

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.861 | 0.615 | 0.340 | 0.438 | 0.811 |
| Random Forest | 0.833 | 0.476 | 0.426 | 0.449 | 0.782 |

## Initial Findings
- Logistic Regression has higher accuracy, precision, and ROC-AUC.
- Random Forest has higher recall and F1-score.
- Attrition detection needs improvement because both models miss many employees who actually leave.
- Accuracy alone is insufficient because the target classes are imbalanced.

## Next Step
Investigate classification thresholds and improve recall while monitoring precision and F1-score.

## Classification Threshold Experiment

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.3 | 0.500 | 0.511 | 0.505 |
| 0.4 | 0.618 | 0.447 | 0.519 |
| 0.5 | 0.615 | 0.340 | 0.438 |
| 0.6 | 0.700 | 0.298 | 0.418 |

### Findings
- Threshold 0.4 achieved the highest F1-score among the tested thresholds.
- Threshold 0.3 achieved the highest recall.
- Threshold 0.6 achieved the highest precision.
- The threshold should be selected according to the business objective.
- Threshold 0.4 is provisional and requires further validation.

## Final Evaluation at Threshold 0.4

| Metric | Score |
|---|---:|
| Accuracy | 86.73% |
| Precision | 61.76% |
| Recall | 44.68% |
| F1-score | 51.85% |
| ROC-AUC | 81.14% |

### Interpretation
- The model achieves 86.73% accuracy on the held-out test set.
- Recall of 44.68% means the model detects fewer than half of actual attrition cases.
- The threshold of 0.4 improves recall and F1-score compared with the default threshold of 0.5.
- These results are educational benchmarks, not evidence of readiness for real HR decisions.