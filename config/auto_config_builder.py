from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import math
import warnings

import numpy as np
import pandas as pd
import yaml

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

from src.drift_guardian.analyzer.regestry.metric_registry import METRIC_REGISTRY
from config.parse_config import REVERSED_THRESHOLD_METRICS
from src.drift_guardian.profiler.baseline_profiler import Profiler


if not METRIC_REGISTRY:
    raise RuntimeError("METRIC_REGISTRY is empty. Cannot infer Metric enum.")


MetricEnum = type(next(iter(METRIC_REGISTRY.keys())))


def _metric(x: str | Any) -> Any:
    """
    Приводит строку к enum Metric из проекта.
    METRIC_REGISTRY имеет ключи типа Metric.
    """
    if isinstance(x, MetricEnum):
        return x
    return MetricEnum(x)


def _metric_name(x: str | Any) -> str:
    x = _metric(x)
    return x.value


REVERSED_THRESHOLD_METRIC_NAMES = {
    _metric_name(metric) for metric in REVERSED_THRESHOLD_METRICS
}


def _is_reversed_threshold_metric(metric_name: str | Any) -> bool:
    return _metric_name(metric_name) in REVERSED_THRESHOLD_METRIC_NAMES


def _is_time_dtype(s: pd.Series) -> bool:
    dtype = s.dtype
    return (
        is_datetime64_any_dtype(dtype)
        or is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    )


# Ограничения взяты из схемы TypedMetricsConfig.
NUMERIC_ONLY_METRICS = {
    "wasserstein_distance",
    "kstest",
}

CATEGORICAL_ONLY_METRICS = {
    "unseen_category_rate",
    "cardinality_ratio",
    "chi2",
    "cramer_v",
    "category_churn",
}

# Разумные дефолты. При желании полностью переопределяются через options.
DEFAULT_NUMERIC_METRICS = [
    "missing_rate",
    "psi",
    "js_divergence",
    "wasserstein_distance",
    "kstest",  # D-statistic, не p-value
]

DEFAULT_CATEGORICAL_METRICS = [
    "missing_rate",
    "psi",
    "js_divergence",
    "unseen_category_rate",
    "cardinality_ratio",
    "chi2",
    "cramer_v",
    "category_churn",
]


ThresholdLike = dict[str, float] | tuple[float, float] | list[float]


def _default_auto_threshold_floors() -> dict[str, ThresholdLike]:
    """
    Минимальные auto thresholds для метрик, у которых при идеальном совпадении
    reference/current bootstrap может дать полностью нулевое распределение.

    Это защищает от бессмысленных конфигов вида:
    missing_rate:
      warning: 0.0
      critical: 1.0e-12

    Значения ниже — именно floors, а не жёсткие thresholds:
    если calibration windows дадут более высокие квантили, будут использованы они.

    Для reversed metrics, например chi2 p_value, эти значения используются
    как direction-aware ограничители, потому что для них меньшее значение хуже:
    warning > critical.

    Для scale-dependent / sample-size-dependent метрик вроде:
    - wasserstein_distance

    дефолтный floor здесь намеренно не задаём, потому что универсальная шкала
    для них может быть некорректной. При необходимости их можно задать вручную
    через options.auto_thresholds.floors.
    """
    return {
        "missing_rate": {
            "warning": 0.005,  # 0.5%
            "critical": 0.02,  # 2%
        },
        "unseen_category_rate": {
            "warning": 0.005,  # 0.5%
            "critical": 0.02,  # 2%
        },
        "category_churn": {
            "warning": 0.005,  # 0.5%
            "critical": 0.02,  # 2%
        },
        "psi": {
            "warning": 0.1,
            "critical": 0.25,
        },
        "js_divergence": {
            "warning": 0.005,
            "critical": 0.02,
        },
        "kstest": {
            "warning": 0.02,
            "critical": 0.05,
        },
        "chi2": {
            "warning": 0.05,
            "critical": 0.01,
        },
        "cramer_v": {
            "warning": 0.02,
            "critical": 0.05,
        },
        "cardinality_ratio": {
            "warning": 0.02,
            "critical": 0.05,
        },
    }


@dataclass
class AutoThresholdSettings:
    enabled: bool = True

    # Как генерировать calibration windows.
    method: Literal["bootstrap", "rolling"] = "bootstrap"

    # Если None, берётся profiler_window_size.
    window_size: int | None = None

    # Кол-во окон для оценки null-distribution метрики.
    n_windows: int = 100

    # Квантили распределения метрики.
    warning_quantile: float = 0.95
    critical_quantile: float = 0.99

    # Добавлять ли feature-level auto thresholds.
    # Если False — автозаполняются только глобальные thresholds.
    per_feature: bool = False

    # Добавлять ли локальные thresholds для prediction_metrics.
    per_prediction: bool = False

    # Минимальные значения thresholds по метрикам.
    # Полезно, если bootstrap даёт везде 0.
    #
    # Для reversed metrics эти значения работают как direction-aware ограничители:
    # например chi2 p_value должен иметь warning > critical.
    #
    # Пример:
    # floors={
    #   "psi": {"warning": 0.1, "critical": 0.25},
    #   "kstest": (0.02, 0.05),
    #   "chi2": {"warning": 0.05, "critical": 0.01},
    # }
    floors: dict[str, ThresholdLike] = field(
        default_factory=_default_auto_threshold_floors
    )

    # Если warning == critical, одна из границ будет чуть сдвинута
    # с учётом направления метрики.
    eps: float = 1e-12

    random_state: int | None = 42

    # Если отдельный расчёт метрики на окне упал.
    # True: warning и пропускаем это окно.
    # False: сразу падаем.
    ignore_metric_errors: bool = True


@dataclass
class ConfigBuildOptions:
    # Явный список фичей. Если None — берём все колонки, кроме prediction_score_column.
    include_columns: list[str] | None = None

    # Колонки, которые не надо мониторить.
    exclude_columns: list[str] = field(default_factory=list)

    # Полностью отключить фичи.
    disabled_features: list[str] = field(default_factory=list)

    # Явно задать типы фичей.
    # {"country": "categorical", "age": "numeric"}
    feature_types: dict[str, Literal["numeric", "categorical"]] = field(default_factory=dict)

    # Глобальные дефолтные наборы метрик по типам.
    numeric_metrics: list[str] | None = None
    categorical_metrics: list[str] | None = None

    # Полностью переопределить metrics для конкретной фичи.
    # {"age": ["missing_rate", "psi", "kstest"]}
    feature_metrics: dict[str, list[str]] = field(default_factory=dict)

    # Отключить метрики везде.
    disabled_metrics: list[str] = field(default_factory=list)

    # Отключить метрики на конкретной фиче.
    # {"country": ["chi2", "cramer_v"]}
    feature_disabled_metrics: dict[str, list[str]] = field(default_factory=dict)

    # Глобальные thresholds.
    # {"psi": {"warning": 0.1, "critical": 0.25}}
    global_thresholds: dict[str, ThresholdLike] = field(default_factory=dict)

    # Локальные thresholds.
    # {"age": {"psi": {"warning": 0.05, "critical": 0.12}}}
    feature_thresholds: dict[str, dict[str, ThresholdLike]] = field(default_factory=dict)

    # Prediction metrics block.
    prediction_enabled: bool = False
    prediction_score_column: str | None = None
    prediction_type: Literal["numeric", "categorical"] | None = None
    prediction_metrics: list[str] | None = None
    prediction_thresholds: dict[str, ThresholdLike] = field(default_factory=dict)

    # Stream drift block напрямую в формате конфига.
    stream_drift: dict[str, ThresholdLike] | None = None

    # Profiler settings.
    profiler_window_size: int = 1000
    merge_threshold: int = 5
    low_cardinality_threshold: int = 15
    profiler_take_sample: bool = True
    sample_float_dtype: str = "float32"
    random_state: int | None = 42

    # Auto-threshold settings.
    auto_thresholds: AutoThresholdSettings = field(default_factory=AutoThresholdSettings)

    # kwargs для конкретных metric functions.
    # {"psi": {"eps": 1e-6}}
    metric_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Если True — ошибка при несовместимой метрике с типом фичи.
    # Если False — несовместимые метрики просто выкидываются.
    strict_metric_compatibility: bool = True

    # Если True — ошибка, если metric отсутствует в METRIC_REGISTRY.
    strict_metric_registry: bool = True

    # Что делать с time columns.
    drop_time_columns: bool = True

    # Что делать с полностью пустыми колонками.
    drop_all_missing_columns: bool = True

    # Что делать, если после фильтрации у фичи не осталось метрик.
    drop_features_without_metrics: bool = True


def build_drift_config(
    df: pd.DataFrame,
    options: ConfigBuildOptions | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Генерирует config dict и при output_path != None сохраняет YAML.

    Возвращает dict, совместимый с твоей pydantic-схемой Config.
    """
    if options is None:
        options = ConfigBuildOptions()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame")

    if df.empty:
        raise ValueError("df is empty")

    _validate_auto_settings(options.auto_thresholds)

    features_meta = _select_and_infer_features(df, options)

    feature_configs: dict[str, dict[str, Any]] = {}
    used_metrics: set[str] = set()

    for col, ftype in features_meta.items():
        metrics = _resolve_metrics_for_feature(col, ftype, options)

        if not metrics:
            if options.drop_features_without_metrics:
                continue
            raise ValueError(f"No metrics left for feature '{col}'")

        local_thresholds = _normalize_thresholds_map(
            options.feature_thresholds.get(col, {})
        )

        feature_block: dict[str, Any] = {
            "type": ftype,
            "metrics": metrics,
        }

        if local_thresholds:
            feature_block["thresholds"] = local_thresholds

        feature_configs[col] = feature_block
        used_metrics.update(metrics)

    prediction_block = _build_prediction_block(df, options)
    if prediction_block.get("enabled", False):
        used_metrics.update(prediction_block["metrics"])

    if not feature_configs and not prediction_block.get("enabled", False):
        raise ValueError("Nothing to monitor: no features and prediction_metrics disabled")

    global_thresholds = _normalize_thresholds_map(options.global_thresholds)

    if options.auto_thresholds.enabled:
        auto_global, auto_feature, auto_prediction = _estimate_missing_thresholds(
            df=df,
            feature_configs=feature_configs,
            prediction_block=prediction_block,
            existing_global_thresholds=global_thresholds,
            options=options,
        )

        # Глобальные auto thresholds нужны, чтобы Config.resolve_feature_thresholds
        # не падал на метриках без локального threshold.
        for metric_name, pair in auto_global.items():
            global_thresholds.setdefault(metric_name, pair)

        if options.auto_thresholds.per_feature:
            for feature, metric_map in auto_feature.items():
                block = feature_configs[feature]
                block.setdefault("thresholds", {})
                for metric_name, pair in metric_map.items():
                    block["thresholds"].setdefault(metric_name, pair)

        if (
            prediction_block.get("enabled", False)
            and options.auto_thresholds.per_prediction
            and auto_prediction
        ):
            prediction_block.setdefault("thresholds", {})
            for metric_name, pair in auto_prediction.items():
                prediction_block["thresholds"].setdefault(metric_name, pair)

    _assert_all_thresholds_present(
        feature_configs=feature_configs,
        prediction_block=prediction_block,
        global_thresholds=global_thresholds,
    )

    config: dict[str, Any] = {
        "features": feature_configs,
        "prediction_metrics": prediction_block,
        "thresholds": global_thresholds,
    }

    stream_drift = _normalize_stream_drift(options.stream_drift)
    if stream_drift:
        config["stream_drift"] = stream_drift

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    return config


def _validate_auto_settings(auto: AutoThresholdSettings) -> None:
    if auto.window_size is not None and auto.window_size <= 0:
        raise ValueError("auto_thresholds.window_size must be positive")

    if auto.n_windows <= 0:
        raise ValueError("auto_thresholds.n_windows must be positive")

    if not (0 < auto.warning_quantile < 1):
        raise ValueError("warning_quantile must be in (0, 1)")

    if not (0 < auto.critical_quantile < 1):
        raise ValueError("critical_quantile must be in (0, 1)")

    if auto.warning_quantile >= auto.critical_quantile:
        raise ValueError("warning_quantile must be less than critical_quantile")

    if auto.eps <= 0:
        raise ValueError("eps must be positive")


def _select_and_infer_features(
    df: pd.DataFrame,
    options: ConfigBuildOptions,
) -> dict[str, Literal["numeric", "categorical"]]:
    if options.include_columns is None:
        columns = list(df.columns)
    else:
        missing = sorted(set(options.include_columns) - set(df.columns))
        if missing:
            raise ValueError(f"include_columns are missing in df: {missing}")
        columns = list(options.include_columns)

    exclude = set(options.exclude_columns) | set(options.disabled_features)

    if options.prediction_score_column is not None:
        # По умолчанию prediction не дублируем как обычную фичу.
        exclude.add(options.prediction_score_column)

    result: dict[str, Literal["numeric", "categorical"]] = {}

    for col in columns:
        if col in exclude:
            continue

        s = df[col]

        if _is_time_dtype(s):
            if options.drop_time_columns:
                continue
            raise ValueError(
                f"Column '{col}' has time dtype '{s.dtype}'. "
                "Profiler does not support time dtypes."
            )

        if s.isna().all():
            if options.drop_all_missing_columns:
                continue
            raise ValueError(f"Column '{col}' contains only missing values")

        if col in options.feature_types:
            result[col] = options.feature_types[col]
            continue

        if is_numeric_dtype(s) and not is_bool_dtype(s):
            result[col] = "numeric"
        else:
            result[col] = "categorical"

    return result


def _resolve_metrics_for_feature(
    feature: str,
    ftype: Literal["numeric", "categorical"],
    options: ConfigBuildOptions,
) -> list[str]:
    if feature in options.feature_metrics:
        raw_metrics = options.feature_metrics[feature]
    elif ftype == "numeric":
        raw_metrics = options.numeric_metrics or DEFAULT_NUMERIC_METRICS
    else:
        raw_metrics = options.categorical_metrics or DEFAULT_CATEGORICAL_METRICS

    disabled = set(options.disabled_metrics)
    disabled.update(options.feature_disabled_metrics.get(feature, []))

    result: list[str] = []

    for m in raw_metrics:
        name = _metric_name(m)

        if name in disabled:
            continue

        if not _is_metric_compatible(name, ftype):
            msg = f"Metric '{name}' is incompatible with feature '{feature}' type '{ftype}'"
            if options.strict_metric_compatibility:
                raise ValueError(msg)
            warnings.warn(msg + ". Dropping metric.", UserWarning)
            continue

        if _metric(name) not in METRIC_REGISTRY:
            msg = f"Metric '{name}' is absent in METRIC_REGISTRY"
            if options.strict_metric_registry:
                raise ValueError(msg)
            warnings.warn(msg + ". Dropping metric.", UserWarning)
            continue

        if name not in result:
            result.append(name)

    return result


def _is_metric_compatible(
    metric_name: str,
    ftype: Literal["numeric", "categorical"],
) -> bool:
    if ftype == "numeric" and metric_name in CATEGORICAL_ONLY_METRICS:
        return False
    if ftype == "categorical" and metric_name in NUMERIC_ONLY_METRICS:
        return False
    return True


def _build_prediction_block(
    df: pd.DataFrame,
    options: ConfigBuildOptions,
) -> dict[str, Any]:
    if not options.prediction_enabled:
        return {"enabled": False}

    score_col = options.prediction_score_column
    if score_col is None:
        raise ValueError("prediction_enabled=True requires prediction_score_column")

    if score_col not in df.columns:
        raise ValueError(f"prediction_score_column '{score_col}' is missing in df")

    s = df[score_col]

    if _is_time_dtype(s):
        raise ValueError(f"prediction_score_column '{score_col}' has unsupported time dtype")

    if s.isna().all():
        raise ValueError(f"prediction_score_column '{score_col}' contains only missing values")

    if options.prediction_type is not None:
        ptype = options.prediction_type
    elif is_numeric_dtype(s) and not is_bool_dtype(s):
        ptype = "numeric"
    else:
        ptype = "categorical"

    if options.prediction_metrics is not None:
        raw_metrics = options.prediction_metrics
    elif ptype == "numeric":
        raw_metrics = ["psi", "kstest"]
    else:
        raw_metrics = ["psi"]

    disabled = set(options.disabled_metrics)

    metrics: list[str] = []
    for m in raw_metrics:
        name = _metric_name(m)

        if name in disabled:
            continue

        if not _is_metric_compatible(name, ptype):
            msg = f"Prediction metric '{name}' is incompatible with type '{ptype}'"
            if options.strict_metric_compatibility:
                raise ValueError(msg)
            warnings.warn(msg + ". Dropping metric.", UserWarning)
            continue

        if _metric(name) not in METRIC_REGISTRY:
            msg = f"Prediction metric '{name}' is absent in METRIC_REGISTRY"
            if options.strict_metric_registry:
                raise ValueError(msg)
            warnings.warn(msg + ". Dropping metric.", UserWarning)
            continue

        if name not in metrics:
            metrics.append(name)

    if not metrics:
        raise ValueError("prediction_metrics enabled, but no metrics left")

    block: dict[str, Any] = {
        "enabled": True,
        "score_column": score_col,
        "type": ptype,
        "metrics": metrics,
    }

    thresholds = _normalize_thresholds_map(options.prediction_thresholds)
    if thresholds:
        block["thresholds"] = thresholds

    return block


def _estimate_missing_thresholds(
    df: pd.DataFrame,
    feature_configs: dict[str, dict[str, Any]],
    prediction_block: dict[str, Any],
    existing_global_thresholds: dict[str, dict[str, float]],
    options: ConfigBuildOptions,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
]:
    """
    Возвращает:
    - auto_global thresholds;
    - auto_feature thresholds;
    - auto_prediction thresholds.
    """
    num_features = [
        feature
        for feature, block in feature_configs.items()
        if block["type"] == "numeric"
    ]
    cat_features = [
        feature
        for feature, block in feature_configs.items()
        if block["type"] == "categorical"
    ]

    prediction_col = None
    if prediction_block.get("enabled", False):
        prediction_col = prediction_block["score_column"]

    profiler = Profiler(
        ref_data=df,
        window_size=options.profiler_window_size,
        num_features=num_features,
        cat_features=cat_features,
        prediction=prediction_col,
        merge_threshold=options.merge_threshold,
        low_cardinality_threshold=options.low_cardinality_threshold,
        take_sample=options.profiler_take_sample,
        sample_float_dtype=options.sample_float_dtype,
        random_state=options.random_state,
    )
    reference_dict = profiler.profile_ref_data()

    window_size = options.auto_thresholds.window_size or options.profiler_window_size
    window_size = min(window_size, len(df))

    global_values: dict[str, list[float]] = {}
    feature_values: dict[str, dict[str, list[float]]] = {
        feature: {m: [] for m in block["metrics"]}
        for feature, block in feature_configs.items()
    }
    prediction_values: dict[str, list[float]] = {}

    if prediction_block.get("enabled", False):
        prediction_values = {m: [] for m in prediction_block["metrics"]}

    windows = _iter_calibration_windows(
        df=df,
        method=options.auto_thresholds.method,
        window_size=window_size,
        n_windows=options.auto_thresholds.n_windows,
        random_state=options.auto_thresholds.random_state,
    )

    for window in windows:
        for feature, block in feature_configs.items():
            for metric_name in block["metrics"]:
                val = _compute_metric_safe(
                    metric_name=metric_name,
                    reference_dict=reference_dict,
                    current=window[feature],
                    options=options,
                    label=f"feature '{feature}'",
                )
                if val is None:
                    continue

                feature_values[feature][metric_name].append(val)
                global_values.setdefault(metric_name, []).append(val)

        if prediction_block.get("enabled", False):
            score_col = prediction_block["score_column"]
            for metric_name in prediction_block["metrics"]:
                val = _compute_metric_safe(
                    metric_name=metric_name,
                    reference_dict=reference_dict,
                    current=window[score_col],
                    options=options,
                    label="prediction_metrics",
                )
                if val is None:
                    continue

                prediction_values[metric_name].append(val)
                global_values.setdefault(metric_name, []).append(val)

    auto_global: dict[str, dict[str, float]] = {}
    auto_feature: dict[str, dict[str, dict[str, float]]] = {}
    auto_prediction: dict[str, dict[str, float]] = {}

    needed_global_metrics = _collect_metrics_requiring_global_threshold(
        feature_configs=feature_configs,
        prediction_block=prediction_block,
        existing_global_thresholds=existing_global_thresholds,
    )

    for metric_name in sorted(needed_global_metrics):
        if metric_name in existing_global_thresholds:
            continue

        vals = global_values.get(metric_name, [])
        auto_global[metric_name] = _threshold_from_values(
            metric_name=metric_name,
            values=vals,
            options=options,
        )

    if options.auto_thresholds.per_feature:
        for feature, metric_map in feature_values.items():
            for metric_name, vals in metric_map.items():
                existing_local = feature_configs[feature].get("thresholds", {})
                if metric_name in existing_local:
                    continue

                auto_feature.setdefault(feature, {})[metric_name] = _threshold_from_values(
                    metric_name=metric_name,
                    values=vals,
                    options=options,
                )

    if prediction_block.get("enabled", False) and options.auto_thresholds.per_prediction:
        existing_local = prediction_block.get("thresholds", {})
        for metric_name, vals in prediction_values.items():
            if metric_name in existing_local:
                continue

            auto_prediction[metric_name] = _threshold_from_values(
                metric_name=metric_name,
                values=vals,
                options=options,
            )

    return auto_global, auto_feature, auto_prediction


def _iter_calibration_windows(
    df: pd.DataFrame,
    method: Literal["bootstrap", "rolling"],
    window_size: int,
    n_windows: int,
    random_state: int | None,
):
    rng = np.random.default_rng(random_state)

    if method == "bootstrap":
        for _ in range(n_windows):
            idx = rng.choice(len(df), size=window_size, replace=True)
            yield df.iloc[idx]

    elif method == "rolling":
        if len(df) <= window_size:
            for _ in range(n_windows):
                yield df
            return

        max_start = len(df) - window_size
        starts = np.linspace(0, max_start, num=n_windows, dtype=int)
        for start in starts:
            yield df.iloc[start : start + window_size]

    else:
        raise ValueError(f"Unknown calibration method: {method}")


def _compute_metric_safe(
    metric_name: str,
    reference_dict: dict[str, Any],
    current: pd.Series,
    options: ConfigBuildOptions,
    label: str,
) -> float | None:
    metric = _metric(metric_name)
    fn = METRIC_REGISTRY[metric]

    kwargs = options.metric_kwargs.get(metric_name, {})

    try:
        value = fn(reference_dict, current, **kwargs)
        value = float(value)

        if not math.isfinite(value):
            return None

        # Drift metrics должны быть неотрицательными.
        # Направление "больше хуже" / "меньше хуже" определяется thresholds.
        # На всякий случай защищаемся от отрицательных чисел.
        if value < 0:
            value = 0.0

        return value

    except Exception as exc:
        msg = f"Metric '{metric_name}' failed for {label}: {exc}"

        if options.auto_thresholds.ignore_metric_errors:
            warnings.warn(msg, UserWarning)
            return None

        raise RuntimeError(msg) from exc


def _collect_metrics_requiring_global_threshold(
    feature_configs: dict[str, dict[str, Any]],
    prediction_block: dict[str, Any],
    existing_global_thresholds: dict[str, dict[str, float]],
) -> set[str]:
    """
    Собирает метрики, которым нужен global threshold.

    Даже если часть фич имеет локальные thresholds, проще и надёжнее
    иметь global fallback для каждой используемой метрики.
    """
    result: set[str] = set(existing_global_thresholds)

    for block in feature_configs.values():
        result.update(block["metrics"])

    if prediction_block.get("enabled", False):
        result.update(prediction_block["metrics"])

    return result


def _threshold_from_values(
    metric_name: str,
    values: list[float],
    options: ConfigBuildOptions,
) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]

    if vals.size == 0:
        raise ValueError(
            f"Cannot auto-estimate thresholds for metric '{metric_name}': "
            "no valid calibration values. Set threshold manually or disable metric."
        )

    is_reversed = _is_reversed_threshold_metric(metric_name)

    if is_reversed:
        # Для reversed metrics меньшее значение хуже.
        #
        # Если для обычных метрик warning_quantile=0.95 и critical_quantile=0.99,
        # то для reversed metrics берём симметричные нижние квантили:
        # warning:  1 - 0.95 = 0.05
        # critical: 1 - 0.99 = 0.01
        warning_quantile = 1.0 - options.auto_thresholds.warning_quantile
        critical_quantile = 1.0 - options.auto_thresholds.critical_quantile

        warning = float(np.quantile(vals, warning_quantile))
        critical = float(np.quantile(vals, critical_quantile))
    else:
        warning = float(np.quantile(vals, options.auto_thresholds.warning_quantile))
        critical = float(np.quantile(vals, options.auto_thresholds.critical_quantile))

    floor = options.auto_thresholds.floors.get(metric_name)
    if floor is not None:
        floor_pair = _normalize_metric_threshold_pair(metric_name, floor)

        if is_reversed:
            # Для reversed metrics меньшее значение хуже.
            #
            # Например chi2 возвращает p_value:
            # warning = 0.05, critical = 0.01.
            #
            # Если bootstrap даёт слишком высокие p_value thresholds вроде:
            # warning = 0.99, critical = 0.97,
            # они будут слишком чувствительными, потому что alert будет срабатывать
            # почти всегда при p_value <= 0.99.
            #
            # Поэтому для reversed metrics используем заданные значения как cap.
            warning = min(warning, floor_pair["warning"])
            critical = min(critical, floor_pair["critical"])
        else:
            # Для обычных метрик большее значение хуже.
            # Floors защищают от бессмысленных нулевых thresholds.
            warning = max(warning, floor_pair["warning"])
            critical = max(critical, floor_pair["critical"])

    warning, critical = _ensure_threshold_direction(
        metric_name=metric_name,
        warning=warning,
        critical=critical,
        eps=options.auto_thresholds.eps,
    )

    return {
        "warning": warning,
        "critical": critical,
    }


def _ensure_threshold_direction(
    metric_name: str,
    warning: float,
    critical: float,
    eps: float,
) -> tuple[float, float]:
    warning = float(warning)
    critical = float(critical)

    if warning < 0:
        warning = 0.0

    if critical < 0:
        critical = 0.0

    if _is_reversed_threshold_metric(metric_name):
        # Для reversed metrics меньшее значение хуже:
        # warning > critical.
        if warning <= critical:
            if warning == 0:
                warning = eps
                critical = 0.0
            else:
                critical = max(
                    warning - max(abs(warning) * 1e-6, eps),
                    0.0,
                )

    else:
        # Для обычных метрик большее значение хуже:
        # warning < critical.
        if warning >= critical:
            if warning == 0:
                critical = eps
            else:
                critical = warning + max(abs(warning) * 1e-6, eps)

    return warning, critical


def _normalize_thresholds_map(
    raw: dict[str, ThresholdLike],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for metric_name, pair in raw.items():
        name = _metric_name(metric_name)
        result[name] = _normalize_metric_threshold_pair(name, pair)

    return result


def _coerce_threshold_pair(pair: ThresholdLike) -> dict[str, float]:
    if isinstance(pair, dict):
        if "warning" not in pair or "critical" not in pair:
            raise ValueError(f"Threshold dict must contain warning and critical: {pair}")

        warning = float(pair["warning"])
        critical = float(pair["critical"])

    elif isinstance(pair, (tuple, list)) and len(pair) == 2:
        warning = float(pair[0])
        critical = float(pair[1])

    else:
        raise ValueError(
            "Threshold must be {'warning': x, 'critical': y} or tuple/list (x, y)"
        )

    if warning < 0 or critical < 0:
        raise ValueError(f"Thresholds must be non-negative: {pair}")

    return {
        "warning": warning,
        "critical": critical,
    }


def _normalize_metric_threshold_pair(
    metric_name: str | Any,
    pair: ThresholdLike,
) -> dict[str, float]:
    name = _metric_name(metric_name)
    result = _coerce_threshold_pair(pair)

    warning = result["warning"]
    critical = result["critical"]

    if _is_reversed_threshold_metric(name):
        if warning <= critical:
            raise ValueError(
                f"For reversed metric '{name}' warning must be greater than critical: {pair}"
            )
    else:
        if warning >= critical:
            raise ValueError(
                f"For metric '{name}' warning must be less than critical: {pair}"
            )

    return result


def _normalize_threshold_pair(pair: ThresholdLike) -> dict[str, float]:
    result = _coerce_threshold_pair(pair)

    warning = result["warning"]
    critical = result["critical"]

    if warning >= critical:
        raise ValueError(f"warning must be less than critical: {pair}")

    return result


def _normalize_stream_drift(
    raw: dict[str, ThresholdLike] | None,
) -> dict[str, dict[str, float]] | None:
    if not raw:
        return None

    result: dict[str, dict[str, float]] = {}
    for name, pair in raw.items():
        result[name] = _normalize_threshold_pair(pair)

    return result


def _assert_all_thresholds_present(
    feature_configs: dict[str, dict[str, Any]],
    prediction_block: dict[str, Any],
    global_thresholds: dict[str, dict[str, float]],
) -> None:
    missing: list[str] = []

    for feature, block in feature_configs.items():
        local = block.get("thresholds", {})
        for metric_name in block["metrics"]:
            if metric_name not in local and metric_name not in global_thresholds:
                missing.append(f"feature '{feature}' / metric '{metric_name}'")

    if prediction_block.get("enabled", False):
        local = prediction_block.get("thresholds", {})
        for metric_name in prediction_block["metrics"]:
            if metric_name not in local and metric_name not in global_thresholds:
                missing.append(f"prediction_metrics / metric '{metric_name}'")

    if missing:
        raise ValueError(
            "Missing thresholds after auto-generation: " + ", ".join(missing)
        )

