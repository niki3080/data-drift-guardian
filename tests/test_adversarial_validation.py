import unittest

import numpy as np
import pandas as pd

from drift_guardian.batch.adversarial_validation import (
    _prepare_features,
    _validate_inputs,
    adversarial_validation,
)


class AdversarialValidationTests(unittest.TestCase):
    def test_returns_auc_and_sorted_feature_importance(self) -> None:
        """Проверяет основной контракт на выборках разного размера."""
        rng = np.random.default_rng(42)
        reference = pd.DataFrame(
            {
                "amount": rng.normal(0.0, 1.0, 120),
                "region": rng.choice(["north", "south", None], 120),
                "active": rng.choice([True, False], 120),
            }
        )
        current = pd.DataFrame(
            {
                "active": rng.choice([True, False], 180),
                "region": rng.choice(["north", "east", None], 180),
                "amount": rng.normal(3.0, 1.0, 180),
            }
        )

        auc, importance = adversarial_validation(
            reference,
            current,
            max_samples=90,
            n_splits=3,
            random_state=42,
        )

        self.assertIsInstance(auc, float)
        self.assertGreater(auc, 0.8)
        self.assertLessEqual(auc, 1.0)
        self.assertListEqual(
            importance.columns.tolist(),
            ["feature", "importance", "importance_std", "rank"],
        )
        self.assertSetEqual(set(importance["feature"]), set(reference.columns))
        self.assertTrue(importance["importance"].is_monotonic_decreasing)
        self.assertListEqual(importance["rank"].tolist(), [1, 2, 3])
        self.assertAlmostEqual(float(importance["importance"].sum()), 1.0)

    def test_prepare_features_uses_reference_dtypes(self) -> None:
        """Проверяет приведение типов и обработку специальных значений."""
        reference = pd.DataFrame(
            {
                "number": pd.Series([1.0, np.nan], dtype="float64"),
                "category": pd.Series(["a", None], dtype="string"),
                "flag": pd.Series([True, False], dtype="bool"),
            }
        )
        features = pd.DataFrame(
            {
                "number": ["1.5", "not-a-number", np.inf],
                "category": ["a", None, "b"],
                "flag": [True, None, False],
            }
        )

        prepared = _prepare_features(
            features,
            reference,
            missing_category="__missing__",
        )

        self.assertEqual(prepared.loc[0, "number"], 1.5)
        self.assertTrue(pd.isna(prepared.loc[1, "number"]))
        self.assertTrue(pd.isna(prepared.loc[2, "number"]))
        self.assertIsInstance(prepared["category"].dtype, pd.CategoricalDtype)
        self.assertIsInstance(prepared["flag"].dtype, pd.CategoricalDtype)
        self.assertEqual(prepared.loc[1, "category"], "__missing__")
        self.assertEqual(prepared.loc[1, "flag"], "__missing__")
        self.assertSetEqual(
            set(prepared["category"].cat.categories),
            {"__missing__", "a", "b"},
        )
        self.assertSetEqual(
            set(prepared["flag"].cat.categories),
            {"__missing__", "False", "True"},
        )

    def test_rejects_invalid_parameter_types(self) -> None:
        """Проверяет понятные ошибки для неверных типов параметров."""
        reference = pd.DataFrame({"value": range(6)})
        current = pd.DataFrame({"value": range(6)})

        invalid_arguments = (
            ({"max_samples": 3.5}, "max_samples must be an integer"),
            ({"max_samples": True}, "max_samples must be an integer"),
            ({"n_splits": 2.5}, "n_splits must be an integer"),
            ({"n_splits": False}, "n_splits must be an integer"),
            ({"missing_category": None}, "missing_category must be a string"),
        )

        for arguments, message in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(TypeError, message):
                    adversarial_validation(reference, current, **arguments)

    def test_rejects_invalid_parameter_values(self) -> None:
        """Проверяет ограничения на значения параметров валидации."""
        reference = pd.DataFrame({"value": range(6)})
        current = pd.DataFrame({"value": range(6)})

        invalid_arguments = (
            ({"max_samples": 0}, "max_samples must be positive"),
            ({"n_splits": 1}, "n_splits must be at least 2"),
            ({"missing_category": ""}, "missing_category must not be empty"),
            ({"max_samples": 2, "n_splits": 3}, "at least 3 rows"),
        )

        for arguments, message in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    adversarial_validation(reference, current, **arguments)

    def test_rejects_different_columns(self) -> None:
        """Не допускает различающийся набор признаков в двух датасетах."""
        reference = pd.DataFrame({"expected": range(6)})
        current = pd.DataFrame({"unexpected": range(6)})

        with self.assertRaisesRegex(ValueError, "must have the same columns"):
            adversarial_validation(reference, current)

    def test_rejects_unsupported_reference_dtype(self) -> None:
        """Явно отклоняет datetime до согласования его преобразования."""
        reference = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=6)}
        )
        current = pd.DataFrame(
            {"timestamp": pd.date_range("2026-02-01", periods=6)}
        )

        with self.assertRaisesRegex(TypeError, "unsupported feature dtype"):
            adversarial_validation(reference, current)

    def test_validate_inputs_accepts_numpy_integer_parameters(self) -> None:
        """Принимает целые NumPy-скаляры, часто встречающиеся в ML-коде."""
        reference = pd.DataFrame({"value": range(6)})
        current = pd.DataFrame({"value": range(6)})

        _validate_inputs(
            reference,
            current,
            max_samples=np.int64(6),
            n_splits=np.int64(3),
            missing_category="__missing__",
        )

    def test_rejects_empty_dataframes(self) -> None:
        """Отклоняет датасеты без строк или без признаков."""
        valid = pd.DataFrame({"value": range(6)})
        empty_inputs = (
            (pd.DataFrame({"value": []}), valid),
            (valid, pd.DataFrame({"value": []})),
            (pd.DataFrame(index=range(6)), pd.DataFrame(index=range(6))),
        )

        for reference, current in empty_inputs:
            with self.subTest(
                reference_shape=reference.shape,
                current_shape=current.shape,
            ):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    adversarial_validation(reference, current)

    def test_constant_features_return_neutral_auc_and_zero_importance(self) -> None:
        """Проверяет детерминированный сценарий без различий и вариативности."""
        reference = pd.DataFrame(
            {
                "number": np.ones(60),
                "category": ["same"] * 60,
            }
        )
        current = reference.copy()

        auc, importance = adversarial_validation(reference, current)

        self.assertAlmostEqual(auc, 0.5)
        self.assertTrue((importance["importance"] == 0.0).all())
        self.assertTrue((importance["importance_std"] == 0.0).all())

    def test_all_missing_numeric_values_are_supported(self) -> None:
        """Проверяет числовой признак, полностью пропущенный в current."""
        reference = pd.DataFrame({"value": np.arange(60, dtype=float)})
        current = pd.DataFrame({"value": [np.nan] * 80})

        auc, importance = adversarial_validation(reference, current)

        self.assertGreater(auc, 0.8)
        self.assertListEqual(importance["feature"].tolist(), ["value"])
        self.assertAlmostEqual(float(importance.loc[0, "importance"]), 1.0)


if __name__ == "__main__":
    unittest.main()
