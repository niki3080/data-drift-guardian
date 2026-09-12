from enum import Enum
from typing import Dict, List, ClassVar

from pydantic import BaseModel, Field, model_validator
import yaml


class FeatureType(str, Enum):
    """Допустимые типы фичей."""

    numeric = "numeric"
    categorical = "categorical"


class Metric(str, Enum):
    missing_rate = "missing_rate"
    psi = "psi"
    mean_zscore = "mean_zscore"
    cramer_v_score = "cramer_v_score"
    unseen_category_rate = "unseen_category_rate"
    cardinality_ratio = "cardinality_ratio"


class PredictionMetric(str, Enum):
    """Метрики для предсказаний."""

    prediction_score_drift = "prediction_score_drift"
    positive_prediction_rate = "positive_prediction_rate"


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


class FeatureConfig(BaseModel):
    type: FeatureType
    metrics: List[Metric] = Field(..., min_length=1)

    NUMERIC_ONLY_METRICS: ClassVar[set[Metric]] = {Metric.mean_zscore}
    CATEGORICAL_ONLY_METRICS: ClassVar[set[Metric]] = {Metric.cramer_v_score}

    @model_validator(mode="after")
    def check_metrics_for_type(self):
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


class PredictionMetricsConfig(BaseModel):
    enabled: bool = True
    score_column: str
    threshold: float = Field(..., ge=0.0, le=1.0)
    metrics: List[PredictionMetric] = Field(..., min_length=1)


# class Thresholds(BaseModel):
#     psi_warning: float = Field(..., ge=0.0)
#     psi_critical: float = Field(..., ge=0.0)
#     zscore_warning: float = Field(..., ge=0.0)
#     zscore_critical: float = Field(..., ge=0.0)
#     unseen_category_rate_warning: float = Field(..., ge=0.0, le=1.0)
#     unseen_category_rate_critical: float = Field(..., ge=0.0, le=1.0)

#     @model_validator(mode="after")
#     def warning_below_critical(self):
#         pairs = [
#             ("psi", self.psi_warning, self.psi_critical),
#             ("zscore", self.zscore_warning, self.zscore_critical),
#             (
#                 "unseen_category_rate",
#                 self.unseen_category_rate_warning,
#                 self.unseen_category_rate_critical,
#             ),
#         ]
#         for name, warn, crit in pairs:
#             if warn >= crit:
#                 raise ValueError(
#                     f"{name}_warning ({warn}) должен быть меньше "
#                     f"{name}_critical ({crit})"
#                 )
#         return self


class Config(BaseModel):
    features: Dict[str, FeatureConfig]
    prediction_metrics: PredictionMetricsConfig
    thresholds: Dict[Metric, ThresholdPair]


def read_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = Config(**raw)
    return config


example_config = read_config(r"F:\s21_proj\data-drift-guardian\config\config.yaml")
print(example_config)