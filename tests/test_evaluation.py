
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def test_holdout_split_has_no_overlapping_rows():
    df = pd.read_csv("data/project1.csv")

    X = df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ],
        errors="ignore",
    )
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    X_dev, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=2026,
        stratify=y,
    )

    assert len(X_dev) + len(X_test) == len(df)
    assert set(X_dev.index).isdisjoint(set(X_test.index))


def test_metrics_are_valid_for_binary_predictions():
    actual = np.array([0, 0, 1, 1, 1, 0])
    predicted = np.array([0, 0, 1, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.4, 0.9, 0.7])

    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted),
        "recall": recall_score(actual, predicted),
        "f1": f1_score(actual, predicted),
        "roc_auc": roc_auc_score(actual, probabilities),
    }

    for name, value in metrics.items():
        assert 0 <= value <= 1, f"{name} is outside [0, 1]"


def test_saved_evaluation_results_are_valid():
    results = pd.read_csv("reports/final_evaluation.csv")

    assert len(results) == 1

    required_metrics = [
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]
    assert set(required_metrics).issubset(results.columns)

    row = results.iloc[0]
    assert 0 <= row["Threshold"] <= 1

    for metric in required_metrics[1:]:
        assert 0 <= row[metric] <= 1
