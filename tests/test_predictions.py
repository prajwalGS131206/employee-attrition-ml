
import numpy as np
import pandas as pd
import pytest

from predict_attrition import predict_employee, load_model_and_threshold


def test_saved_model_and_threshold_load():
    model, threshold = load_model_and_threshold()

    assert hasattr(model, "predict_proba")
    assert 0 <= threshold <= 1


def test_prediction_probability_and_label():
    model, threshold = load_model_and_threshold()
    df = pd.read_csv("data/project1.csv")

    excluded = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    employee = df.drop(columns=excluded, errors="ignore").iloc[[0]]

    probability, prediction = predict_employee(
        model, employee, threshold
    )

    assert 0 <= probability <= 1
    assert prediction in {"Yes", "No"}
    assert prediction == ("Yes" if probability >= threshold else "No")


def test_invalid_threshold_is_rejected():
    model, _ = load_model_and_threshold()
    df = pd.read_csv("data/project1.csv")
    employee = df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ],
        errors="ignore",
    ).iloc[[0]]

    with pytest.raises(ValueError):
        predict_employee(model, employee, 1.5)


def test_dataset_has_expected_target_values():
    df = pd.read_csv("data/project1.csv")

    assert not df.empty
    assert "Attrition" in df.columns
    assert df["Attrition"].notna().all()
    assert set(df["Attrition"].unique()) == {"Yes", "No"}


def test_dataset_has_no_duplicate_rows():
    df = pd.read_csv("data/project1.csv")

    assert not df.duplicated().any()


def test_probabilities_are_finite():
    model, threshold = load_model_and_threshold()
    df = pd.read_csv("data/project1.csv")

    excluded = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    employee = df.drop(columns=excluded, errors="ignore").iloc[[0]]

    probability, _ = predict_employee(model, employee, threshold)

    assert np.isfinite(probability)
