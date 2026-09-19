from datetime import datetime

import pandas as pd
import pytest

from src.drift_guardian.data_quality_checker.checker import SchemaChecker


@pytest.fixture
def reference_df():
    return pd.DataFrame(
        {
            "user_id": pd.Series([1, 2], dtype="int64"),
            "score": pd.Series([1.5, 2.5], dtype="float64"),
            "name": pd.Series(["Alice", "Bob"], dtype="object"),
            "is_active": pd.Series([True, False], dtype="bool"),
            "event_time": pd.to_datetime(
                ["2026-09-19 08:00:00", "2026-09-19 08:01:00"]
            ),
        }
    )


@pytest.fixture
def checker(reference_df):
    return SchemaChecker(
        reference_df=reference_df,
        required_cols={"user_id", "score", "event_time"},
    )


def test_check_event_valid_event(checker):
    event_time = datetime(2026, 9, 19, 8, 0, 0)

    event = {
        "user_id": 1,
        "score": 10.5,
        "name": "Alice",
        "is_active": True,
        "event_time": event_time,
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is True
    assert error is None
    assert validated == event


def test_check_event_missing_optional_column_is_valid(checker):
    event = {
        "user_id": 1,
        "score": 10.5,
        "event_time": datetime(2026, 9, 19, 8, 0, 0),
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is True
    assert error is None
    assert validated["user_id"] == 1
    assert validated["score"] == 10.5
    assert validated["event_time"] == datetime(2026, 9, 19, 8, 0, 0)

    # optional-поля есть в model_dump, но заполнены None
    assert validated["name"] is None
    assert validated["is_active"] is None


def test_check_event_missing_required_column_is_invalid(checker):
    event = {
        "score": 10.5,
        "name": "Alice",
        "is_active": True,
        "event_time": datetime(2026, 9, 19, 8, 0, 0),
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is False
    assert validated is None
    assert "Missing REQUIRED columns" in error
    assert "user_id" in error


def test_check_event_invalid_type_is_invalid(checker):
    event = {
        "user_id": "not-an-int",
        "score": 10.5,
        "name": "Alice",
        "is_active": True,
        "event_time": datetime(2026, 9, 19, 8, 0, 0),
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is False
    assert validated is None
    assert error is not None


def test_check_event_extra_column_is_ignored_by_pydantic_model(checker):
    event = {
        "user_id": 1,
        "score": 10.5,
        "name": "Alice",
        "is_active": True,
        "event_time": datetime(2026, 9, 19, 8, 0, 0),
        "unknown_field": "extra",
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is True
    assert error is None
    assert "unknown_field" not in validated


def test_check_df_valid_dataframe_passes(checker, reference_df):
    checker.check_df(reference_df)


def test_check_df_missing_optional_column_passes(checker, reference_df):
    df = reference_df.drop(columns=["name"])

    checker.check_df(df)


def test_check_df_missing_required_column_raises(checker, reference_df):
    df = reference_df.drop(columns=["user_id"])

    with pytest.raises(ValueError, match="Missing REQUIRED columns"):
        checker.check_df(df)


def test_check_df_required_dtype_mismatch_raises(checker, reference_df):
    df = reference_df.copy()
    df["user_id"] = df["user_id"].astype("float64")

    with pytest.raises(ValueError, match="dtype mismatch"):
        checker.check_df(df)


def test_check_df_optional_dtype_mismatch_does_not_raise(checker, reference_df):
    df = reference_df.copy()
    df["name"] = pd.Series([1, 2], dtype="int64")

    checker.check_df(df)


def test_check_df_required_dtype_mismatch_does_not_raise_when_disabled(reference_df):
    checker = SchemaChecker(
        reference_df=reference_df,
        required_cols={"user_id"},
        raise_on_missing_required=False,
    )

    df = reference_df.copy()
    df["user_id"] = df["user_id"].astype("float64")

    checker.check_df(df)


def test_check_df_missing_required_does_not_raise_when_disabled(reference_df):
    checker = SchemaChecker(
        reference_df=reference_df,
        required_cols={"user_id"},
        raise_on_missing_required=False,
    )

    df = reference_df.drop(columns=["user_id"])

    checker.check_df(df)


def test_init_raises_for_nullable_int_dtype():
    reference_df = pd.DataFrame(
        {
            "user_id": pd.Series([1, 2], dtype="Int64"),
        }
    )

    with pytest.raises(ValueError, match="unsupported nullable dtype"):
        SchemaChecker(reference_df=reference_df)


def test_init_raises_for_nullable_boolean_dtype():
    reference_df = pd.DataFrame(
        {
            "flag": pd.Series([True, False], dtype="boolean"),
        }
    )

    with pytest.raises(ValueError, match="unsupported nullable dtype"):
        SchemaChecker(reference_df=reference_df)


def test_init_raises_for_datetime_column_not_equal_to_time_column():
    reference_df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                ["2026-09-19 08:00:00", "2026-09-19 08:01:00"]
            ),
        }
    )

    with pytest.raises(ValueError, match="only the configured"):
        SchemaChecker(reference_df=reference_df)


def test_init_allows_datetime_column_equal_to_time_column():
    reference_df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                ["2026-09-19 08:00:00", "2026-09-19 08:01:00"]
            ),
        }
    )

    checker = SchemaChecker(reference_df=reference_df, time_column="created_at")

    event = {
        "created_at": datetime(2026, 9, 19, 8, 0, 0),
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is True
    assert error is None
    assert validated["created_at"] == datetime(2026, 9, 19, 8, 0, 0)


def test_init_raises_for_unmapped_dtype():
    reference_df = pd.DataFrame(
        {
            "period_col": pd.period_range("2026-01", periods=2, freq="M"),
        }
    )

    with pytest.raises(ValueError, match="unmapped dtype"):
        SchemaChecker(reference_df=reference_df)


def test_category_dtype_is_treated_as_str():
    reference_df = pd.DataFrame(
        {
            "segment": pd.Series(["a", "b"], dtype="category"),
        }
    )

    checker = SchemaChecker(reference_df=reference_df)

    is_valid, validated, error = checker.check_event({"segment": "a"})

    assert is_valid is True
    assert error is None
    assert validated["segment"] == "a"
