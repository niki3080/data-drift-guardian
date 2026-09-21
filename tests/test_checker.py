from datetime import datetime

import pandas as pd
import pytest

from drift_guardian.data_quality_checker.checker import SchemaChecker


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Возвращает базовый reference DataFrame для тестов схемы."""
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
def checker(reference_df: pd.DataFrame) -> SchemaChecker:
    """Создаёт checker с обязательными полями realtime-события."""
    return SchemaChecker(
        reference_df=reference_df,
        required_cols={"user_id", "score", "event_time"},
    )


def test_check_event_valid_event(checker: SchemaChecker) -> None:
    event = {
        "user_id": 1,
        "score": 10.5,
        "name": "Alice",
        "is_active": True,
        "event_time": datetime(2026, 9, 19, 8, 0, 0),
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is True
    assert error is None
    assert validated == event


def test_check_event_missing_optional_column_is_valid(checker: SchemaChecker) -> None:
    event = {
        "user_id": 1,
        "score": 10.5,
        "event_time": datetime(2026, 9, 19, 8, 0, 0),
    }

    is_valid, validated, error = checker.check_event(event)

    assert is_valid is True
    assert error is None
    assert validated is not None
    assert validated["name"] is None
    assert validated["is_active"] is None


def test_check_event_missing_required_column_is_invalid(checker: SchemaChecker) -> None:
    is_valid, validated, error = checker.check_event(
        {
            "score": 10.5,
            "event_time": datetime(2026, 9, 19, 8, 0, 0),
        }
    )

    assert is_valid is False
    assert validated is None
    assert error is not None
    assert "Missing REQUIRED columns" in error
    assert "user_id" in error


def test_check_event_invalid_type_is_invalid(checker: SchemaChecker) -> None:
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


def test_check_event_extra_column_is_ignored(checker: SchemaChecker) -> None:
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
    assert validated is not None
    assert "unknown_field" not in validated


def test_check_df_valid_dataframe_passes(
    checker: SchemaChecker,
    reference_df: pd.DataFrame,
) -> None:
    checker.check_df(reference_df)


def test_check_df_missing_optional_column_passes(
    checker: SchemaChecker,
    reference_df: pd.DataFrame,
) -> None:
    checker.check_df(reference_df.drop(columns=["name"]))


def test_check_df_missing_required_column_raises(
    checker: SchemaChecker,
    reference_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Missing REQUIRED columns"):
        checker.check_df(reference_df.drop(columns=["user_id"]))


def test_check_df_required_dtype_mismatch_raises(
    checker: SchemaChecker,
    reference_df: pd.DataFrame,
) -> None:
    current = reference_df.copy()
    current["user_id"] = current["user_id"].astype("float64")

    with pytest.raises(ValueError, match="dtype mismatch"):
        checker.check_df(current)


def test_check_df_optional_dtype_mismatch_does_not_raise(
    checker: SchemaChecker,
    reference_df: pd.DataFrame,
) -> None:
    current = reference_df.copy()
    current["name"] = pd.Series([1, 2], dtype="int64")
    checker.check_df(current)


@pytest.mark.parametrize("dtype", ["string", "category"])
def test_string_like_dtype_is_supported_for_events(dtype: str) -> None:
    reference = pd.DataFrame({"segment": pd.Series(["a", "b"], dtype=dtype)})
    checker = SchemaChecker(reference_df=reference)

    is_valid, validated, error = checker.check_event({"segment": "a"})

    assert is_valid is True
    assert error is None
    assert validated is not None
    assert validated["segment"] == "a"


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
def test_nullable_dtype_is_rejected(dtype: str) -> None:
    values = [1, 2] if dtype == "Int64" else [True, False]
    reference = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})

    with pytest.raises(ValueError, match="unsupported nullable dtype"):
        SchemaChecker(reference_df=reference)


def test_datetime_is_allowed_only_for_configured_time_column() -> None:
    reference = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                ["2026-09-19 08:00:00", "2026-09-19 08:01:00"]
            )
        }
    )

    with pytest.raises(ValueError, match="only the configured"):
        SchemaChecker(reference_df=reference)

    checker = SchemaChecker(reference_df=reference, time_column="created_at")
    is_valid, _, error = checker.check_event(
        {"created_at": datetime(2026, 9, 19, 8, 0, 0)}
    )

    assert is_valid is True
    assert error is None


def test_required_validation_can_be_non_fatal(reference_df: pd.DataFrame) -> None:
    checker = SchemaChecker(
        reference_df=reference_df,
        required_cols={"user_id"},
        raise_on_missing_required=False,
    )

    wrong_dtype = reference_df.copy()
    wrong_dtype["user_id"] = wrong_dtype["user_id"].astype("float64")
    checker.check_df(wrong_dtype)
    checker.check_df(reference_df.drop(columns=["user_id"]))


def test_unmapped_period_dtype_is_rejected() -> None:
    reference = pd.DataFrame(
        {"period": pd.period_range("2026-01", periods=2, freq="M")}
    )

    with pytest.raises(ValueError, match="unmapped dtype"):
        SchemaChecker(reference_df=reference)


def test_category_df_allows_different_category_values() -> None:
    reference = pd.DataFrame(
        {
            "segment": pd.Series(
                pd.Categorical(["a", "b"], categories=["a", "b"])
            )
        }
    )
    current = pd.DataFrame(
        {
            "segment": pd.Series(
                pd.Categorical(["b", "c"], categories=["b", "c"])
            )
        }
    )
    checker = SchemaChecker(
        reference_df=reference,
        required_cols={"segment"},
    )

    checker.check_df(current)
