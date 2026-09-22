from enum import StrEnum
from typing import ClassVar, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


REVERSED_THRESHOLD_METRICS: set[Metric] = {Metric.chi2}


def _validate_metric_threshold(
    metric: Metric,
    pair: ThresholdPair,
    label: str,
) -> None:
    if metric in REVERSED_THRESHOLD_METRICS:
        if pair.warning <= pair.critical:
            raise ValueError(
                f"For {label}.{metric.value}, warning ({pair.warning}) must be "
                f"greater than critical ({pair.critical})"
            )
        return

    if pair.warning >= pair.critical:
        raise ValueError(
            f"For {label}.{metric.value}, warning ({pair.warning}) must be "
            f"less than critical ({pair.critical})"
        )


class TypedMetricsConfig(BaseModel):
    """Базовая схема метрик для numeric и categorical признаков."""

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

        if self.type == FeatureType.categorical and (
            wrong := set(self.metrics) & self.NUMERIC_ONLY_METRICS
        ):
            raise ValueError(
                f"{[m.value for m in wrong]} are not allowed for categorical features"
            )
        if self.type == FeatureType.numeric and (
            wrong := set(self.metrics) & self.CATEGORICAL_ONLY_METRICS
        ):
            raise ValueError(
                f"{[m.value for m in wrong]} are not allowed for numeric features"
            )
        return self

    @model_validator(mode="after")
    def check_thresholds_are_used_metrics(self):
        if self.metrics and self.thresholds:
            extra = set(self.thresholds) - set(self.metrics)
            if extra:
                raise ValueError(
                    f"Thresholds are configured for metrics {[m.value for m in extra]} "
                    "that are not enabled for this feature"
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
                f"When enabled=True, required fields are missing: {', '.join(missing)}"
            )
        return self


class StreamDriftConfig(BaseModel):
    """Пороговые настройки технических метрик realtime-потока."""

    model_config = ConfigDict(extra="forbid")

    drift_stream_status: Optional[ThresholdPair] = None
    drift_event_time_lag_seconds: Optional[ThresholdPair] = None
    drift_window_time_span_seconds: Optional[ThresholdPair] = None
    drift_max_event_gap_seconds: Optional[ThresholdPair] = None
    drift_invalid_event_time_rate: Optional[ThresholdPair] = None
    drift_late_event_rate: Optional[ThresholdPair] = None

    @model_validator(mode="after")
    def validate_thresholds(self):
        for name, pair in self.__dict__.items():
            if isinstance(pair, ThresholdPair) and pair.warning >= pair.critical:
                raise ValueError(
                    f"For stream_drift.{name}, warning ({pair.warning}) must be "
                    f"less than critical ({pair.critical})"
                )
        return self


class AdversarialValidationConfig(BaseModel):
    """Настройки adversarial validation и порогов ROC-AUC."""

    enabled: bool = True
    thresholds: Optional[ThresholdPair] = None

    @model_validator(mode="after")
    def validate_auc_thresholds(self):
        pair = self.thresholds
        if pair is None:
            return self
        if not (0.5 <= pair.warning < pair.critical <= 1.0):
            raise ValueError(
                "adversarial_validation.thresholds must satisfy "
                "0.5 <= warning < critical <= 1.0"
            )
        return self


class Config(BaseModel):
    features: Dict[str, FeatureConfig]
    prediction_metrics: PredictionMetricsConfig
    stream_drift: Optional[StreamDriftConfig] = None
    adversarial_validation: Optional[AdversarialValidationConfig] = None

    # глобальные пороги используются, если для feature нет локального override
    thresholds: Dict[Metric, ThresholdPair] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metric_thresholds(self):
        for metric, pair in self.thresholds.items():
            _validate_metric_threshold(metric, pair, "thresholds")

        for name, feature in self.features.items():
            for metric, pair in feature.thresholds.items():
                _validate_metric_threshold(metric, pair, f"feature '{name}'")

        for metric, pair in self.prediction_metrics.thresholds.items():
            _validate_metric_threshold(metric, pair, "prediction_metrics")

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
                    f"No thresholds are configured for {label} metrics {missing} "
                    "in either Config.thresholds or the local feature settings"
                )
            block.resolved_thresholds = merged

        for name, feature in self.features.items():
            resolve(feature, f"feature '{name}'")

        if self.prediction_metrics.enabled:
            resolve(self.prediction_metrics, "prediction_metrics")

        return self


def read_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = Config(**raw)
    return config
