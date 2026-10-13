
# Employee Attrition — Initial EDA Findings

## Objective
Explore employee HR data to understand patterns associated with employee attrition before building a predictive model.

## Dataset
IBM HR Analytics Employee Attrition dataset.

## Initial Findings

### 1. Attrition distribution
- Employees who stayed: 83.88%
- Employees who left: 16.12%

The target is imbalanced. Accuracy alone may be misleading when evaluating future classification models.

### 2. Monthly income
- Average monthly income for employees who stayed: 6832.74
- Average monthly income for employees who left: 4787.09

In this dataset, employees who left had a lower average monthly income than employees who stayed.

This is an observed association, not proof that income causes attrition. Further analysis is needed to examine job role, seniority, overtime, and other factors.

## Visualizations
- `figures/attrition_distribution.png`
- `figures/income_by_attrition.png`

## Limitations
- These are descriptive findings, not causal conclusions.
- The dataset may not represent the workforce of every organization.
- Predictive performance has not yet been evaluated.

## Next Steps
1. Validate data quality and feature types.
2. Investigate attrition across relevant employee groups.
3. Establish a baseline machine learning model.
4. Evaluate precision, recall, F1-score, and the confusion matrix.
