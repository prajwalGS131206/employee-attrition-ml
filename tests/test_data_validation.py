
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/project1.csv")


def load_dataset():
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


def test_dataset_is_not_empty():
    df = load_dataset()
    assert len(df) > 0


def test_dataset_has_expected_target():
    df = load_dataset()

    assert "Attrition" in df.columns
    assert df["Attrition"].notna().all()
    assert set(df["Attrition"].unique()) == {"Yes", "No"}


def test_dataset_has_no_missing_values():
    df = load_dataset()
    assert df.isna().sum().sum() == 0


def test_dataset_has_no_duplicate_rows():
    df = load_dataset()
    assert not df.duplicated().any()


def test_identifier_and_constant_columns_are_not_model_features():
    df = load_dataset()

    excluded = {
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    }

    features = df.drop(columns=list(excluded), errors="ignore")

    assert "Attrition" not in features.columns
    assert "EmployeeNumber" not in features.columns
    assert features.shape[1] == 30


def test_target_has_both_classes():
    df = load_dataset()

    counts = df["Attrition"].value_counts()

    assert counts["Yes"] > 0
    assert counts["No"] > 0
