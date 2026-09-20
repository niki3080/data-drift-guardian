from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

from drift_guardian.data_quality_checker.checker import SchemaChecker


class SchemaCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_df = pd.DataFrame(
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
        self.checker = SchemaChecker(
            reference_df=self.reference_df,
            required_cols={"user_id", "score", "event_time"},
        )

    def test_check_event_valid_event(self) -> None:
        event = {
            "user_id": 1,
            "score": 10.5,
            "name": "Alice",
            "is_active": True,
            "event_time": datetime(2026, 9, 19, 8, 0, 0),
        }

        is_valid, validated, error = self.checker.check_event(event)

        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertEqual(validated, event)

    def test_check_event_missing_optional_column_is_valid(self) -> None:
        event = {
            "user_id": 1,
            "score": 10.5,
            "event_time": datetime(2026, 9, 19, 8, 0, 0),
        }

        is_valid, validated, error = self.checker.check_event(event)

        self.assertTrue(is_valid)
        self.assertIsNone(error)
        assert validated is not None
        self.assertIsNone(validated["name"])
        self.assertIsNone(validated["is_active"])

    def test_check_event_missing_required_column_is_invalid(self) -> None:
        is_valid, validated, error = self.checker.check_event(
            {
                "score": 10.5,
                "event_time": datetime(2026, 9, 19, 8, 0, 0),
            }
        )

        self.assertFalse(is_valid)
        self.assertIsNone(validated)
        self.assertIn("Missing REQUIRED columns", error or "")
        self.assertIn("user_id", error or "")

    def test_check_df_valid_dataframe_passes(self) -> None:
        self.checker.check_df(self.reference_df)

    def test_check_df_missing_optional_column_passes(self) -> None:
        self.checker.check_df(self.reference_df.drop(columns=["name"]))

    def test_check_df_missing_required_column_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing REQUIRED columns"):
            self.checker.check_df(self.reference_df.drop(columns=["user_id"]))

    def test_check_df_required_dtype_mismatch_raises(self) -> None:
        current = self.reference_df.copy()
        current["user_id"] = current["user_id"].astype("float64")

        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            self.checker.check_df(current)

    def test_check_df_optional_dtype_mismatch_does_not_raise(self) -> None:
        current = self.reference_df.copy()
        current["name"] = pd.Series([1, 2], dtype="int64")
        self.checker.check_df(current)

    def test_pandas_string_dtype_is_supported(self) -> None:
        reference = pd.DataFrame(
            {"segment": pd.Series(["a", "b"], dtype="string")}
        )
        checker = SchemaChecker(reference_df=reference)

        is_valid, validated, error = checker.check_event({"segment": "a"})

        self.assertTrue(is_valid)
        self.assertIsNone(error)
        assert validated is not None
        self.assertEqual(validated["segment"], "a")

    def test_nullable_integer_dtype_is_rejected(self) -> None:
        reference = pd.DataFrame(
            {"user_id": pd.Series([1, 2], dtype="Int64")}
        )

        with self.assertRaisesRegex(ValueError, "unsupported nullable dtype"):
            SchemaChecker(reference_df=reference)

    def test_datetime_is_allowed_only_for_configured_time_column(self) -> None:
        reference = pd.DataFrame(
            {
                "created_at": pd.to_datetime(
                    ["2026-09-19 08:00:00", "2026-09-19 08:01:00"]
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "only the configured"):
            SchemaChecker(reference_df=reference)

        checker = SchemaChecker(reference_df=reference, time_column="created_at")
        is_valid, _, error = checker.check_event(
            {"created_at": datetime(2026, 9, 19, 8, 0, 0)}
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_check_event_invalid_type_is_invalid(self) -> None:
        event = {
            "user_id": "not-an-int",
            "score": 10.5,
            "name": "Alice",
            "is_active": True,
            "event_time": datetime(2026, 9, 19, 8, 0, 0),
        }

        is_valid, validated, error = self.checker.check_event(event)

        self.assertFalse(is_valid)
        self.assertIsNone(validated)
        self.assertIsNotNone(error)

    def test_check_event_extra_column_is_ignored(self) -> None:
        event = {
            "user_id": 1,
            "score": 10.5,
            "name": "Alice",
            "is_active": True,
            "event_time": datetime(2026, 9, 19, 8, 0, 0),
            "unknown_field": "extra",
        }

        is_valid, validated, error = self.checker.check_event(event)

        self.assertTrue(is_valid)
        self.assertIsNone(error)
        assert validated is not None
        self.assertNotIn("unknown_field", validated)

    def test_check_df_required_dtype_mismatch_can_be_non_fatal(self) -> None:
        checker = SchemaChecker(
            reference_df=self.reference_df,
            required_cols={"user_id"},
            raise_on_missing_required=False,
        )
        current = self.reference_df.copy()
        current["user_id"] = current["user_id"].astype("float64")

        checker.check_df(current)

    def test_check_df_missing_required_can_be_non_fatal(self) -> None:
        checker = SchemaChecker(
            reference_df=self.reference_df,
            required_cols={"user_id"},
            raise_on_missing_required=False,
        )

        checker.check_df(self.reference_df.drop(columns=["user_id"]))

    def test_nullable_boolean_dtype_is_rejected(self) -> None:
        reference = pd.DataFrame(
            {"flag": pd.Series([True, False], dtype="boolean")}
        )

        with self.assertRaisesRegex(ValueError, "unsupported nullable dtype"):
            SchemaChecker(reference_df=reference)

    def test_unmapped_period_dtype_is_rejected(self) -> None:
        reference = pd.DataFrame(
            {"period": pd.period_range("2026-01", periods=2, freq="M")}
        )

        with self.assertRaisesRegex(ValueError, "unmapped dtype"):
            SchemaChecker(reference_df=reference)

    def test_category_dtype_is_supported_for_events(self) -> None:
        reference = pd.DataFrame(
            {"segment": pd.Series(["a", "b"], dtype="category")}
        )
        checker = SchemaChecker(reference_df=reference)

        is_valid, validated, error = checker.check_event({"segment": "a"})

        self.assertTrue(is_valid)
        self.assertIsNone(error)
        assert validated is not None
        self.assertEqual(validated["segment"], "a")

    def test_category_df_allows_different_category_values(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
