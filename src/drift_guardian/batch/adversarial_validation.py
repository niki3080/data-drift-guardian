"""Dataset-level drift detection with adversarial validation."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


__all__ = ["adversarial_validation"]

MAX_SAMPLES = 100_000
N_SPLITS = 3
RANDOM_STATE = 42
MISSING_CATEGORY = "__missing__"


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    max_samples: int = MAX_SAMPLES,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    missing_category: str = MISSING_CATEGORY,
) -> tuple[float, pd.DataFrame]:
    """Обучить LightGBM отличать reference от current.

    Чем выше AUC, тем сильнее дрифт. Два датасета уменьшаются до размера
    меньшего из них, но не более чем ``max_samples`` строк на датасет, чтобы
    классы были сбалансированы 1:1. ROC AUC считается на out-of-fold
    предсказаниях кросс-валидации.

    Схема признаков берется из ``reference``. Числовые пропущенные значения
    сохраняются как NaN для нативной обработки алгоритмом LightGBM. Столбцы
    типов object, string, category и boolean обрабатываются как категориальные,
    их пропущенные значения обозначаются как ``missing_category``.

    Args:
        reference: Базовый (эталонный) датасет. Его dtypes определяют схему признаков.
        current: Текущий датасет для сравнения с базовым.
        max_samples: Верхняя граница размера каждой из двух выборок.
        n_splits: Число фолдов кросс-валидации (минимум 2).
        random_state: Сид для сэмплирования, разбиения на фолды и обучения модели.
        missing_category: Строка, которой заполняются пропуски в категориальных колонках.

    Returns:
        tuple[float, pd.DataFrame]: OOF ROC AUC и таблица feature importance
        со столбцами ``feature``, ``importance``, ``importance_std`` и ``rank``.
        Importance – это gain LightGBM, нормализованный и усредненный по фолдам.

    Raises:
        TypeError: Если аргументы имеют неподдерживаемые типы.
        ValueError: Если значения аргументов не подходят для валидации.
    """

    # Проверка входных данных до сэмплирования и обучения модели.
    _validate_inputs(
        reference,
        current,
        max_samples=max_samples,
        n_splits=n_splits,
        missing_category=missing_category,
    )

    sample_size = min(len(reference), len(current), max_samples)

    # Выравнивание классов путем случайной выборки из reference и current
    reference_sample = reference.sample(n=sample_size, random_state=random_state)
    current_sample = current.loc[:, reference.columns].sample(
        n=sample_size,
        random_state=random_state,
    )

    # Объединение выборок в один DataFrame для обучения модели
    features = pd.concat([reference_sample, current_sample], ignore_index=True)

    # Подготовка признаков: приведение типов к типам reference, обработка пропусков
    features = _prepare_features(
        features, reference, missing_category=missing_category
    )

    # Создание целевой переменной: 0 для reference, 1 для current
    target = np.concatenate(
        [np.zeros(sample_size, dtype=np.int8), np.ones(sample_size, dtype=np.int8)]
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probabilities = np.zeros(len(features), dtype=float)
    fold_importances: list[np.ndarray] = []

    # Обучение модели и сбор метрик
    for train_indices, validation_indices in splitter.split(features, target):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=100,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            importance_type="gain",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(features.iloc[train_indices], target[train_indices])

        oof_probabilities[validation_indices] = model.predict_proba(
            features.iloc[validation_indices]
        )[:, 1]

        gain = model.booster_.feature_importance(importance_type="gain")
        gain_sum = gain.sum()
        normalized_gain = gain / gain_sum if gain_sum else np.zeros_like(gain)
        fold_importances.append(normalized_gain)

    roc_auc = float(roc_auc_score(target, oof_probabilities))
    importance_values = np.vstack(fold_importances)
    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": importance_values.mean(axis=0),
            "importance_std": importance_values.std(axis=0),
        }
    )
    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)

    return roc_auc, feature_importance


def _validate_inputs(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    max_samples: int,
    n_splits: int,
    missing_category: str,
) -> None:
    """Проверка входных данных на корректность."""
    # Типы основных аргументов
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise TypeError("reference and current must be pandas DataFrames")
    if reference.empty or current.empty:
        raise ValueError("reference and current must not be empty")
    if reference.columns.has_duplicates or current.columns.has_duplicates:
        raise ValueError("reference and current must not contain duplicate columns")
    if not all(isinstance(column, str) for column in reference.columns):
        raise TypeError("all feature names must be strings")

    # Типы дополнительных аргументов
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral):
        raise TypeError("max_samples must be an integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise TypeError("n_splits must be an integer")
    if not isinstance(missing_category, str):
        raise TypeError("missing_category must be a string")

    # Допустимые значения дополнительных аргументов
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if not missing_category:
        raise ValueError("missing_category must not be empty")

    # Проверка соответствия колонок reference и current
    reference_columns = set(reference.columns)
    current_columns = set(current.columns)
    if reference_columns != current_columns:
        missing = sorted(reference_columns - current_columns)
        unexpected = sorted(current_columns - reference_columns)
        raise ValueError(
            "reference and current must have the same columns; "
            f"missing in current: {missing}; unexpected in current: {unexpected}"
        )

    # Проверка достаточности размера выборок для кросс-валидации
    sample_size = min(len(reference), len(current), max_samples)
    if sample_size < n_splits:
        raise ValueError(
            f"each dataset must contain at least {n_splits} rows for "
            f"{n_splits}-fold validation"
        )


def _prepare_features(
    features: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    missing_category: str,
) -> pd.DataFrame:
    """Определить типы по reference и подготовить признаки для LightGBM."""
    prepared = features.copy()

    for column in reference.columns:
        dtype = reference[column].dtype

        if is_numeric_dtype(dtype) and not is_bool_dtype(dtype) and dtype.kind != "c":
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
            continue

        if (
            is_bool_dtype(dtype)
            or is_object_dtype(dtype)
            or is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            prepared[column] = (
                prepared[column]
                .astype("string")
                .fillna(missing_category)
                .astype("category")
            )
            continue

        raise TypeError(f"unsupported feature dtype: {column} ({dtype})")

    return prepared
