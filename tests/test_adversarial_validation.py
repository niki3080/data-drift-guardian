from typing import Any

import numpy as np
import pandas as pd
import pytest

from drift_guardian.analyzer.methods.batch.adversarial_validation import (
    _prepare_features,
    _validate_inputs,
    adversarial_validation,
)


ADVERSARIAL_VALIDATION_MODULE = (
    "drift_guardian.analyzer.methods.batch.adversarial_validation"
)


@pytest.fixture()
def simple_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Минимальные одинаковые reference/current для проверки валидации параметров."""
    reference = pd.DataFrame({"value": range(6)})
    current = pd.DataFrame({"value": range(6)})
    return reference, current


@pytest.fixture()
def shifted_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Выборки с заметным сдвигом распределения для adversarial validation."""
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
            "unused": rng.normal(size=180),
            "active": rng.choice([True, False], 180),
            "region": rng.choice(["north", "east", None], 180),
            "amount": rng.normal(3.0, 1.0, 180),
        }
    )

    return reference, current


@pytest.fixture()
def prepare_features_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Данные для проверки приведения типов и обработки специальных значений."""
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

    return reference, features


@pytest.fixture()
def datetime_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference/current с datetime-признаком."""
    reference = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=6)}
    )
    current = pd.DataFrame(
        {"timestamp": pd.date_range("2026-02-01", periods=6)}
    )
    return reference, current


@pytest.fixture()
def constant_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Одинаковые константные выборки без различий и вариативности."""
    reference = pd.DataFrame(
        {
            "number": np.ones(60),
            "category": ["same"] * 60,
        }
    )
    current = reference.copy()
    return reference, current


@pytest.fixture()
def all_missing_numeric_current_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Числовой признак, полностью пропущенный в current."""
    reference = pd.DataFrame({"value": np.arange(60, dtype=float)})
    current = pd.DataFrame({"value": [np.nan] * 80})
    return reference, current


@pytest.fixture()
def fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Подменяет LightGBM-классификатор и возвращает созданные fake-инстансы."""
    instances = []

    class FakeBooster:
        @staticmethod
        def feature_importance(*, importance_type: str) -> np.ndarray:
            assert importance_type == "gain"
            return np.array([1.0])

    class FakeClassifier:
        def __init__(self, **params) -> None:
            self.params = params
            self.best_iteration_ = 7
            self.booster_ = FakeBooster()
            self.fit_kwargs = None
            instances.append(self)

        def fit(self, X, y, **kwargs) -> None:
            self.fit_kwargs = kwargs

        def predict_proba(self, X, **kwargs) -> np.ndarray:
            assert kwargs["num_iteration"] == 7
            return np.tile([0.5, 0.5], (len(X), 1))

    monkeypatch.setattr(
        f"{ADVERSARIAL_VALIDATION_MODULE}.LGBMClassifier",
        FakeClassifier,
    )
    monkeypatch.setattr(
        f"{ADVERSARIAL_VALIDATION_MODULE}.early_stopping",
        lambda *args, **kwargs: "callback",
    )

    return instances


def test_returns_auc_and_sorted_feature_importance(
    shifted_reference_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Проверяет основной контракт на выборках разного размера."""
    reference, current = shifted_reference_current

    auc, importance = adversarial_validation(
        reference,
        current,
        max_samples=90,
        n_splits=3,
        random_state=42,
    )

    assert isinstance(auc, float)
    assert auc > 0.8
    assert auc <= 1.0

    assert importance.columns.tolist() == [
        "feature",
        "importance",
        "importance_std",
        "rank",
    ]
    assert set(importance["feature"]) == set(reference.columns)
    assert importance["importance"].is_monotonic_decreasing
    assert importance["rank"].tolist() == [1, 2, 3]
    assert float(importance["importance"].sum()) == pytest.approx(1.0)
    for attr in (
        "roc_auc_cv_mean",
        "roc_auc_cv_std",
        "roc_auc_cv_min",
        "roc_auc_cv_max",
        "driver_consistency",
    ):
        assert attr in importance.attrs

    assert 0.0 <= importance.attrs["driver_consistency"] <= 1.0


def test_prepare_features_uses_reference_dtypes(
    prepare_features_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Проверяет приведение типов и обработку специальных значений."""
    reference, features = prepare_features_data

    prepared = _prepare_features(
        features,
        reference,
        missing_category="__missing__",
    )

    assert prepared.loc[0, "number"] == 1.5
    assert pd.isna(prepared.loc[1, "number"])
    assert pd.isna(prepared.loc[2, "number"])

    assert isinstance(prepared["category"].dtype, pd.CategoricalDtype)
    assert isinstance(prepared["flag"].dtype, pd.CategoricalDtype)

    assert prepared.loc[1, "category"] == "__missing__"
    assert prepared.loc[1, "flag"] == "__missing__"

    assert set(prepared["category"].cat.categories) == {
        "__missing__",
        "a",
        "b",
    }
    assert set(prepared["flag"].cat.categories) == {
        "__missing__",
        "False",
        "True",
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_samples": 3.5}, "max_samples must be an integer"),
        ({"max_samples": True}, "max_samples must be an integer"),
        ({"n_splits": 2.5}, "n_splits must be an integer"),
        ({"n_splits": False}, "n_splits must be an integer"),
        ({"missing_category": None}, "missing_category must be a string"),
        (
            {"lightgbm_params": []},
            "lightgbm_params must be a dictionary or None",
        ),
    ],
)
def test_rejects_invalid_parameter_types(
    simple_reference_current: tuple[pd.DataFrame, pd.DataFrame],
    arguments: dict[str, Any],
    message: str,
) -> None:
    """Проверяет понятные ошибки для неверных типов параметров."""
    reference, current = simple_reference_current

    with pytest.raises(TypeError, match=message):
        adversarial_validation(reference, current, **arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_samples": 0}, "max_samples must be positive"),
        ({"n_splits": 1}, "n_splits must be at least 2"),
        ({"missing_category": ""}, "missing_category must not be empty"),
        ({"max_samples": 2, "n_splits": 3}, "at least 3 rows"),
    ],
)
def test_rejects_invalid_parameter_values(
    simple_reference_current: tuple[pd.DataFrame, pd.DataFrame],
    arguments: dict[str, Any],
    message: str,
) -> None:
    """Проверяет ограничения на значения параметров валидации."""
    reference, current = simple_reference_current

    with pytest.raises(ValueError, match=message):
        adversarial_validation(reference, current, **arguments)


def test_rejects_unsupported_reference_dtype(
    datetime_reference_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Явно отклоняет datetime до согласования его преобразования."""
    reference, current = datetime_reference_current

    with pytest.raises(TypeError, match="unsupported feature dtype"):
        adversarial_validation(reference, current)


def test_validate_inputs_accepts_numpy_integer_parameters(
    simple_reference_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Принимает целые NumPy-скаляры, часто встречающиеся в ML-коде."""
    reference, current = simple_reference_current

    _validate_inputs(
        reference,
        current,
        max_samples=np.int64(6),
        n_splits=np.int64(3),
        missing_category="__missing__",
        lightgbm_params=None,
    )


def test_constant_features_return_neutral_auc_and_zero_importance(
    constant_reference_current: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Проверяет детерминированный сценарий без различий и вариативности."""
    reference, current = constant_reference_current

    auc, importance = adversarial_validation(reference, current)

    assert auc == pytest.approx(0.5)
    assert (importance["importance"] == 0.0).all()
    assert (importance["importance_std"] == 0.0).all()


def test_all_missing_numeric_values_are_supported(
    all_missing_numeric_current_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Проверяет числовой признак, полностью пропущенный в current."""
    reference, current = all_missing_numeric_current_data

    auc, importance = adversarial_validation(reference, current)

    assert auc > 0.8
    assert importance["feature"].tolist() == ["value"]
    assert float(importance.loc[0, "importance"]) == pytest.approx(1.0)


def test_lightgbm_params_are_merged_and_invariants_are_preserved(
    simple_reference_current: tuple[pd.DataFrame, pd.DataFrame],
    fake_lightgbm: list[Any],
) -> None:
    """Проверяет передачу параметров модели и обязательные objective/metric."""
    reference, current = simple_reference_current
    instances = fake_lightgbm

    adversarial_validation(
        reference,
        current,
        lightgbm_params={
            "n_estimators": 25,
            "learning_rate": 0.2,
            "objective": "multiclass",
            "metric": "binary_logloss",
        },
    )

    assert len(instances) == 3

    for model in instances:
        assert model.params["n_estimators"] == 25
        assert model.params["learning_rate"] == 0.2
        assert model.params["objective"] == "binary"
        assert model.params["metric"] == "auc"
        assert "eval_metric" not in model.fit_kwargs
        assert "eval_X" not in model.fit_kwargs
        assert "eval_y" not in model.fit_kwargs
        assert "eval_set" in model.fit_kwargs
        assert model.fit_kwargs["callbacks"] == ["callback"]
