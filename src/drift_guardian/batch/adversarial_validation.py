"""Dataset-level drift detection with adversarial validation."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    max_samples: int = 100_000,
    n_splits: int = 3,
    random_state: int = 42,
    missing_category: str = "__missing__",
    lightgbm_params: dict[str, Any] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Оценить разделимость подготовленных датасетов reference и current.

    Функция обучает LightGBM различать строки двух датасетов и возвращает
    ROC AUC на out-of-fold предсказаниях. Чем выше AUC, тем легче модели
    различить датасеты. Метрика служит индикатором возможного дрифта.

    Для баланса классов каждый датасет сэмплируется до размера меньшего
    из них, но не более `max_samples` строк.

    Схема признаков берется из `reference`. Числовые пропущенные значения
    сохраняются как NaN для нативной обработки алгоритмом LightGBM. 
    Столбцы типов object, string, category и boolean обрабатываются как 
    категориальные, их пропущенные значения обозначаются как `missing_category`.

    Args:
        reference: Базовый (эталонный) датасет. Его dtypes определяют схему признаков.
        current: Текущий датасет для сравнения с базовым. Должен содержать
            как минимум все колонки reference (лишние колонки игнорируются).
        max_samples: Верхняя граница размера каждой из двух выборок.
        n_splits: Число фолдов кросс-валидации (минимум 2).
        random_state: Сид для сэмплирования, разбиения на фолды и обучения модели.
        missing_category: Строка, которой заполняются пропуски в категориальных колонках.
        lightgbm_params: Дополнительные параметры `LGBMClassifier`.
            Перезаписывают стандартные параметры модели, кроме `objective="binary"`
            и `metric="auc"`. Feature importance всегда рассчитывается по `gain`.

    Returns:
        tuple[float, pd.DataFrame]: OOF ROC AUC и таблица feature importance
            со столбцами `feature`, `importance`, `importance_std` и `rank`.
            Importance – это gain LightGBM, нормализованный и усредненный по фолдам.

    Raises:
        TypeError: Если аргументы имеют неподдерживаемые типы.
        ValueError: Если значения аргументов не подходят для валидации.

    Note:
        В каждом фолде early stopping выбирает число деревьев по той же
        валидационной выборке, для которой строятся OOF-предсказания.
        Это может завысить итоговый ROC AUC. Полученная оценка используется
        как индикатор возможного дрифта.
    """

    # Проверка параметров функции до сэмплирования и обучения модели
    _validate_inputs(
        reference,
        current,
        max_samples=max_samples,
        n_splits=n_splits,
        missing_category=missing_category,
        lightgbm_params=lightgbm_params,
    )

    # Выравнивание классов путем случайной выборки из reference и current
    sample_size = min(len(reference), len(current), max_samples)

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

    # Настройка кросс-валидации
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probabilities = np.zeros(len(features), dtype=float)
    fold_importances: list[np.ndarray] = []

    # Настройка параметров LightGBM с возможностью переопределения пользователем
    default_model_params = {
        "n_estimators": 1_000,
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "importance_type": "gain",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model_params = default_model_params | (lightgbm_params or {})

    # Invariants
    model_params["objective"] = "binary"
    model_params["metric"] = "auc"

    # Обучение модели и сбор метрик по фолдам
    for train_indices, validation_indices in splitter.split(features, target):
        # Разделение на обучающую и валидационную выборки
        X_train = features.iloc[train_indices]
        y_train = target[train_indices]

        X_val = features.iloc[validation_indices]
        y_val = target[validation_indices]

        model = LGBMClassifier(**model_params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(
                    stopping_rounds=50,
                    verbose=False,
                )
            ],
        )

        oof_probabilities[validation_indices] = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_,
        )[:, 1]

        # Сбор feature importances по фолдам
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_sum = gain.sum()

        # Защита от деления на ноль, если модель не использовала ни одного признака
        normalized_gain = gain / gain_sum if gain_sum else np.zeros_like(gain)
        fold_importances.append(normalized_gain)

    # Вычисление ROC AUC и усредненной feature importance
    roc_auc = float(roc_auc_score(target, oof_probabilities))
    importance_values = np.vstack(fold_importances)
    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": importance_values.mean(axis=0),
            "importance_std": importance_values.std(axis=0),
        }
    )
    # Сортировка по убыванию важности и присвоение рангов
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
    lightgbm_params: dict[str, Any] | None,
) -> None:
    """Проверка дополнительных аргументов на корректность."""
    # Типы дополнительных аргументов
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral):
        raise TypeError("max_samples must be an integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise TypeError("n_splits must be an integer")
    if not isinstance(missing_category, str):
        raise TypeError("missing_category must be a string")
    if lightgbm_params is not None and not isinstance(lightgbm_params, dict):
        raise TypeError("lightgbm_params must be a dictionary or None")

    # Допустимые значения дополнительных аргументов
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if not missing_category:
        raise ValueError("missing_category must not be empty")

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
