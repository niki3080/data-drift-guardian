"""Adversarial validation для поиска drift между reference и current."""

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
    """Оценивает различимость reference и current через LightGBM.

    Датасеты выравниваются по размеру, после чего модель учится отличать
    reference от current. Итоговый ROC-AUC считается по out-of-fold
    предсказаниям, а feature importance усредняется по CV-фолдам.

    Схема признаков берётся из reference. Числовые пропуски сохраняются как
    NaN, категориальные заполняются значением ``missing_category``.

    :param reference: Эталонный DataFrame, определяющий набор и типы признаков.
    :param current: Текущий DataFrame. Лишние колонки игнорируются.
    :param max_samples: Максимальное число строк каждого класса после балансировки.
    :param n_splits: Число CV-фолдов, минимум 2.
    :param random_state: Seed для sampling, CV и LightGBM.
    :param missing_category: Значение для пропусков категориальных признаков.
    :param lightgbm_params: Дополнительные параметры ``LGBMClassifier``.
    :return: OOF ROC-AUC и DataFrame с feature importance.
    :raises TypeError: если переданы неподдерживаемые типы аргументов.
    :raises ValueError: если параметры или размеры выборок некорректны.
    """

    # проверяем параметры до sampling и обучения модели
    _validate_inputs(
        reference,
        current,
        max_samples=max_samples,
        n_splits=n_splits,
        missing_category=missing_category,
        lightgbm_params=lightgbm_params,
    )

    # выравниваем классы случайной выборкой из reference и current
    sample_size = min(len(reference), len(current), max_samples)

    reference_sample = reference.sample(n=sample_size, random_state=random_state)
    current_sample = current.loc[:, reference.columns].sample(
        n=sample_size,
        random_state=random_state,
    )

    # объединяем reference и current для обучения бинарного классификатора
    features = pd.concat([reference_sample, current_sample], ignore_index=True)

    # приводим признаки к схеме reference и обрабатываем пропуски
    features = _prepare_features(
        features, reference, missing_category=missing_category
    )

    # target: 0 для reference, 1 для current
    target = np.concatenate(
        [np.zeros(sample_size, dtype=np.int8), np.ones(sample_size, dtype=np.int8)]
    )

    # создаём воспроизводимое stratified CV-разбиение
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probabilities = np.zeros(len(features), dtype=float)
    fold_importances: list[np.ndarray] = []
    fold_roc_auc_scores: list[float] = []

    # задаём базовые параметры LightGBM и применяем пользовательские overrides
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

    # objective и metric фиксированы контрактом AV
    model_params["objective"] = "binary"
    model_params["metric"] = "auc"

    # обучаем модель и собираем OOF-предсказания по фолдам
    for train_indices, validation_indices in splitter.split(features, target):
        # разделяем текущий фолд на train и validation
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

        fold_probabilities = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_,
        )[:, 1]
        oof_probabilities[validation_indices] = fold_probabilities
        fold_roc_auc_scores.append(float(roc_auc_score(y_val, fold_probabilities)))

        # сохраняем нормализованный gain текущего фолда
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_sum = gain.sum()

        # нулевой gain оставляем нулевым вектором importance
        normalized_gain = gain / gain_sum if gain_sum else np.zeros_like(gain)
        fold_importances.append(normalized_gain)

    # считаем общий OOF ROC-AUC и агрегируем importance по фолдам
    roc_auc = float(roc_auc_score(target, oof_probabilities))
    importance_values = np.vstack(fold_importances)
    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": importance_values.mean(axis=0),
            "importance_std": importance_values.std(axis=0),
        }
    )
    # сортируем признаки по убыванию importance и добавляем rank
    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)
    # сохраняем дополнительные CV-метрики в attrs, не меняя публичный return contract
    fold_auc = np.asarray(fold_roc_auc_scores, dtype=float)
    feature_importance.attrs["roc_auc_cv_mean"] = float(fold_auc.mean())
    feature_importance.attrs["roc_auc_cv_std"] = float(fold_auc.std())
    feature_importance.attrs["roc_auc_cv_min"] = float(fold_auc.min())
    feature_importance.attrs["roc_auc_cv_max"] = float(fold_auc.max())
    feature_importance.attrs["driver_consistency"] = _mean_pairwise_cosine(
        importance_values
    )

    return roc_auc, feature_importance


def _mean_pairwise_cosine(vectors: np.ndarray) -> float:
    """Считает согласованность feature importance между CV-фолдами.

    Для каждой пары фолдов используется cosine similarity нормализованных
    gain-векторов. Два нулевых вектора считаются согласованными.

    :param vectors: Матрица importance, одна строка на CV-фолд.
    :return: Среднее pairwise cosine similarity в диапазоне от 0 до 1.
    """
    if len(vectors) < 2:
        return 1.0

    similarities: list[float] = []
    for left_idx in range(len(vectors) - 1):
        left = np.asarray(vectors[left_idx], dtype=float)
        for right_idx in range(left_idx + 1, len(vectors)):
            right = np.asarray(vectors[right_idx], dtype=float)
            left_norm = float(np.linalg.norm(left))
            right_norm = float(np.linalg.norm(right))
            if left_norm == 0.0 and right_norm == 0.0:
                similarities.append(1.0)
            elif left_norm == 0.0 or right_norm == 0.0:
                similarities.append(0.0)
            else:
                similarities.append(
                    float(np.dot(left, right) / (left_norm * right_norm))
                )

    return float(np.mean(similarities))


def _validate_inputs(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    max_samples: int,
    n_splits: int,
    missing_category: str,
    lightgbm_params: dict[str, Any] | None,
) -> None:
    """Проверяет параметры adversarial validation.

    :raises TypeError: если тип аргумента не поддерживается.
    :raises ValueError: если значение аргумента или размер выборки некорректны.
    """
    # проверяем типы дополнительных аргументов
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral):
        raise TypeError("max_samples must be an integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise TypeError("n_splits must be an integer")
    if not isinstance(missing_category, str):
        raise TypeError("missing_category must be a string")
    if lightgbm_params is not None and not isinstance(lightgbm_params, dict):
        raise TypeError("lightgbm_params must be a dictionary or None")

    # проверяем допустимые значения дополнительных аргументов
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if not missing_category:
        raise ValueError("missing_category must not be empty")

    # проверяем, что после балансировки данных достаточно для CV
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
    """Приводит признаки к схеме reference для обучения LightGBM.

    :param features: Объединённый DataFrame reference и current.
    :param reference: DataFrame, определяющий ожидаемые dtypes.
    :param missing_category: Значение для категориальных пропусков.
    :return: Подготовленный DataFrame признаков.
    :raises TypeError: если dtype признака не поддерживается.
    """
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
