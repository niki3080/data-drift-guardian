from typing import Dict, List, ClassVar, Optional
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator
import yaml


class FeatureType(StrEnum):
    numeric = "numeric"
    categorical = "categorical"


class Metric(StrEnum):
    missing_rate = "missing_rate"
    psi = "psi"
    unseen_category_rate = "unseen_category_rate"
    cardinality_ratio = "cardinality_ratio"
    js_divergence = "js_divergence"
    wasserstein_distance = "wasserstein_distance"
    chi2 = "chi2"
    cramer_v = "cramer_v"
    category_churn = "category_churn"
    kstest = "kstest"


# Метрики, для которых чем меньше значение, тем хуже.
# Например, chi2 у вас возвращает p_value:
# warning = 0.05, critical = 0.01
#
# То есть для этих метрик должно выполняться:
# warning > critical
REVERSED_THRESHOLD_METRICS: set[Metric] = {
    Metric.chi2,
}


class ThresholdPair(BaseModel):
    warning: float = Field(..., ge=0.0)
    critical: float = Field(..., ge=0.0)


def validate_threshold_direction(
    metric: Metric,
    pair: ThresholdPair,
    label: str,
) -> None:
    """
    Проверяет корректность направления warning/critical.

    Для обычных метрик:
        warning < critical

    Для метрик из REVERSED_THRESHOLD_METRICS:
        warning > critical
    """

    if metric in REVERSED_THRESHOLD_METRICS:
        if pair.warning <= pair.critical:
            raise ValueError(
                f"Для {label}.{metric.value} warning ({pair.warning}) "
                f"должен быть больше critical ({pair.critical})"
            )
    else:
        if pair.warning >= pair.critical:
            raise ValueError(
                f"Для {label}.{metric.value} warning ({pair.warning}) "
                f"должен быть меньше critical ({pair.critical})"
            )


class TypedMetricsConfig(BaseModel):
    """
    Базовый класс для конфигов, где набор допустимых метрик
    зависит от типа фичи: numeric / categorical.
    """

    type: Optional[FeatureType] = None
    metrics: List[Metric] = Field(default_factory=list)

    thresholds: Dict[Metric, ThresholdPair] = Field(default_factory=dict)

    resolved_thresholds: Dict[Metric, ThresholdPair] = Field(default_factory=dict)

    NUMERIC_ONLY_METRICS: ClassVar[set[Metric]] = {
        Metric.wasserstein_distance,
        Metric.kstest,
    }

    CATEGORICAL_ONLY_METRICS: ClassVar[set[Metric]] = {
        Metric.unseen_category_rate,
        Metric.cardinality_ratio,
        Metric.chi2,
        Metric.cramer_v,
        Metric.category_churn,
    }

    @model_validator(mode="after")
    def check_metrics_for_type(self):
        if self.type is None or not self.metrics:
            return self

        if self.type == FeatureType.categorical:
            wrong = set(self.metrics) & self.NUMERIC_ONLY_METRICS

            if wrong:
                raise ValueError(
                    f"{[m.value for m in wrong]} "
                    "недопустим(ы) для categorical-фичей"
                )

        if self.type == FeatureType.numeric:
            wrong = set(self.metrics) & self.CATEGORICAL_ONLY_METRICS

            if wrong:
                raise ValueError(
                    f"{[m.value for m in wrong]} "
                    "недопустим(ы) для numeric-фичей"
                )

        return self

    @model_validator(mode="after")
    def check_thresholds_are_used_metrics(self):
        if self.metrics and self.thresholds:
            extra = set(self.thresholds) - set(self.metrics)

            if extra:
                raise ValueError(
                    f"thresholds заданы для метрик {[m.value for m in extra]}, "
                    "которые не входят в metrics этой фичи"
                )

        return self

    @model_validator(mode="after")
    def check_local_threshold_directions(self):
        for metric, pair in self.thresholds.items():
            validate_threshold_direction(
                metric=metric,
                pair=pair,
                label="local_thresholds",
            )

        return self


class FeatureConfig(TypedMetricsConfig):
    type: FeatureType
    metrics: List[Metric] = Field(..., min_length=1)


class PredictionMetricsConfig(TypedMetricsConfig):
    enabled: bool = True
    score_column: Optional[str] = None

    @model_validator(mode="after")
    def check_required_when_enabled(self):
        if not self.enabled:
            return self

        missing = []

        if self.type is None:
            missing.append("type")

        if not self.metrics:
            missing.append("metrics")

        if self.score_column is None:
            missing.append("score_column")

        if missing:
            raise ValueError(
                f"При enabled=True обязательны поля: {', '.join(missing)}"
            )

        return self


class StreamDriftConfig(BaseModel):
    """
    Метрики состояния стрима событий: event-time drift monitoring.

    Каждая метрика опциональна, но если задана — обязана содержать
    пару warning/critical.

    Для stream_drift используется обычная логика:
        warning < critical
    """

    drift_stream_status: Optional[ThresholdPair] = None
    drift_event_time_lag_seconds: Optional[ThresholdPair] = None
    drift_window_time_span_seconds: Optional[ThresholdPair] = None
    drift_max_event_gap_seconds: Optional[ThresholdPair] = None
    drift_invalid_event_time_rate: Optional[ThresholdPair] = None
    # Window-local доля late-событий для realtime stream health.
    # Lifetime counters сохраняются отдельно для observability и совместимости.
    drift_late_event_rate: Optional[ThresholdPair] = None
    drift_late_events_total: Optional[ThresholdPair] = None
    drift_out_of_order_events_total: Optional[ThresholdPair] = None

    @model_validator(mode="after")
    def check_stream_drift_threshold_directions(self):
        for field_name in self.__class__.model_fields:
            pair = getattr(self, field_name)

            if pair is None:
                continue

            if pair.warning >= pair.critical:
                raise ValueError(
                    f"Для stream_drift.{field_name} warning ({pair.warning}) "
                    f"должен быть меньше critical ({pair.critical})"
                )

        return self


class LightGBMConfig(BaseModel):
    """
    Часто используемые гиперпараметры LightGBM в sklearn API
    (LGBMClassifier / LGBMRegressor).

    Метод `to_params()` возвращает словарь без None-значений,
    готовый для распаковки: LGBMClassifier(**cfg.to_params())
    """

    n_estimators: int = 1_000
    learning_rate: float = 0.05
    max_depth: int = 4
    num_leaves: int = 15
    importance_type: str = 'gain'
    min_child_samples: int = 20
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_jobs: int = -1
    random_state: Optional[int] = None
    class_weight: Optional[str] = None
    objective: Optional[str] = None
    boosting_type: str = "gbdt"
    verbosity: int = -1

    def to_params(self) -> dict:
        return {
            k: v
            for k, v in self.model_dump().items()
            if v is not None
        }


class AdversarialValidationConfig(BaseModel):
    """
    Настройки adversarial validation: периодический запуск проверки
    на сравнение распределений (например, train vs recent data)
    с помощью бинарного классификатора.
    """

    enabled: bool = False

    # Периодичность запуска adversarial validation, в минутах
    interval_minutes: Optional[int] = Field(default=None, gt=0)

    max_samples: Optional[int] = Field(default=100_000, gt=0)
    n_splits: Optional[int] = Field(default=3, ge=2)
    random_state: Optional[int] = 42
    missing_category: Optional[str] = "__missing__"

    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)

    @model_validator(mode="after")
    def check_required_when_enabled(self):
        if not self.enabled:
            return self

        if self.interval_minutes is None:
            raise ValueError(
                "При adversarial_validation.enabled=True обязателен "
                "interval_minutes (периодичность запуска в минутах)"
            )

        return self


class Config(BaseModel):
    features: Dict[str, FeatureConfig]
    prediction_metrics: PredictionMetricsConfig
    stream_drift: Optional[StreamDriftConfig] = None
    adversarial_validation: AdversarialValidationConfig = Field(
        default_factory=AdversarialValidationConfig
    )

    # Глобальные дефолтные thresholds по метрике.
    # Используются как fallback, если у конкретной фичи нет override.
    thresholds: Dict[Metric, ThresholdPair] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_global_threshold_directions(self):
        for metric, pair in self.thresholds.items():
            validate_threshold_direction(
                metric=metric,
                pair=pair,
                label="global_thresholds",
            )

        return self

    @model_validator(mode="after")
    def resolve_feature_thresholds(self):
        def resolve(block: TypedMetricsConfig, label: str):
            merged: Dict[Metric, ThresholdPair] = {}
            missing: List[str] = []

            for metric in block.metrics:
                if metric in block.thresholds:
                    merged[metric] = block.thresholds[metric]
                elif metric in self.thresholds:
                    merged[metric] = self.thresholds[metric]
                else:
                    missing.append(metric.value)

            if missing:
                raise ValueError(
                    f"Для {label} не заданы thresholds для метрик {missing} "
                    "(ни глобально в Config.thresholds, ни локально в самой фиче)"
                )

            block.resolved_thresholds = merged

        for name, feature in self.features.items():
            resolve(feature, f"feature '{name}'")

        if self.prediction_metrics.enabled:
            resolve(self.prediction_metrics, "prediction_metrics")

        return self


def read_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def config_from_dict(raw: dict) -> Config:
    return Config(**raw)
