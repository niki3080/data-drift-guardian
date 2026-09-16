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
    quantile_drift = "quantile_drift"
    chi2 = "chi2"
    cramer_v = "cramer_v"
    category_churn = "category_churn"
    kstest = "kstest"


class ThresholdPair(BaseModel):
    warning: float = Field(..., ge=0.0)
    critical: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def warning_below_critical(self):
        if self.warning >= self.critical:
            raise ValueError(
                f"warning ({self.warning}) должен быть меньше "
                f"critical ({self.critical})"
            )
        return self


class TypedMetricsConfig(BaseModel):
    """
    Базовый класс для конфигов, где набор допустимых метрик
    зависит от типа фичи (numeric / categorical).

    Поля type/metrics здесь опциональны, так как в некоторых
    наследниках (например PredictionMetricsConfig) они нужны
    только при определённых условиях (enabled=True).
    Наследники, которым эти поля нужны всегда (FeatureConfig),
    переобъявляют их как обязательные.
    """

    type: Optional[FeatureType] = None
    metrics: List[Metric] = Field(default_factory=list)

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

        if self.type == FeatureType.categorical and (
            wrong := set(self.metrics) & self.NUMERIC_ONLY_METRICS
        ):
            raise ValueError(
                f"{[m.value for m in wrong]} недопустим(ы) для categorical-фичей"
            )
        if self.type == FeatureType.numeric and (
            wrong := set(self.metrics) & self.CATEGORICAL_ONLY_METRICS
        ):
            raise ValueError(
                f"{[m.value for m in wrong]} недопустим(ы) для numeric-фичей"
            )
        return self


class FeatureConfig(TypedMetricsConfig):
    """
    Конфиг метрик для обычной фичи.
    Здесь type и metrics обязательны всегда.
    """

    type: FeatureType
    metrics: List[Metric] = Field(..., min_length=1)


class PredictionMetricsConfig(TypedMetricsConfig):
    """
    Конфиг метрик для предсказаний.

    type / metrics / score_column обязательны только когда enabled=True.
    При enabled=False достаточно указать только enabled: false.
    """

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


class Config(BaseModel):
    features: Dict[str, FeatureConfig]
    prediction_metrics: PredictionMetricsConfig
    thresholds: Dict[Metric, ThresholdPair]


def read_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = Config(**raw)
    return config



# example_config = read_config(r"F:\s21_proj\data-drift-guardian\config\config.yaml")
# print(example_config)