from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Iterable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Info,
    ProcessCollector,
    disable_created_metrics,
    start_http_server,
)

from drift_guardian.ingestion.stream_metrics import StreamSnapshot

STATUS_TO_NUMBER = {
    "insufficient_data": -1,
    "ok": 0,
    "passed": 0,
    "warning": 1,
    "critical": 2,
    "error": 2,
}

PREDICTION_METRIC_ALIASES = {
    "psi": "prediction_psi",
    "prediction_score_drift": "prediction_psi",
}


disable_created_metrics()


# ------------------------------------------------------------------ #
# Преобразование значений drift-report
# ------------------------------------------------------------------ #
def _status_number(value: Any, default: int = -1) -> int:
    """Преобразует текстовый или числовой status в Prometheus-код."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if value is None:
        return default
    return STATUS_TO_NUMBER.get(str(value).lower(), default)


def _finite_float(value: Any) -> float | None:
    """Возвращает конечное float-значение либо None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_timestamp_seconds(value: Any) -> float | None:
    """Преобразует Unix/ISO-8601 timestamp в секунды Unix epoch."""
    numeric = _finite_float(value)
    if numeric is not None and not isinstance(value, str):
        return numeric
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.timestamp()


def _alert_status(metric: str, alerts: Iterable[str]) -> str | None:
    """Восстанавливает status legacy-метрики по списку alerts."""
    alerts_set = {str(alert) for alert in alerts}
    if f"{metric}_critical" in alerts_set:
        return "critical"
    if f"{metric}_warning" in alerts_set:
        return "warning"
    return None


# ------------------------------------------------------------------ #
# Экспорт метрик в Prometheus
# ------------------------------------------------------------------ #
class PrometheusExporter:
    """Экспортирует drift-report и технические метрики потока в Prometheus.

    Основной контракт drift-метрик:
      - drift_status_feature{feature,type}
      - drift_metric_value{feature,type,metric}
      - drift_status{feature,type,metric}
      - drift_threshold{metric,level}
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Создаёт изолированный набор Prometheus collectors и state streaks."""
        self.registry = registry or CollectorRegistry()
        # Dashboard использует process_start_time_seconds для uptime exporter.
        ProcessCollector(registry=self.registry)

        self.overall_status = Gauge(
            "drift_overall_status",
            "Overall drift status: -1=insufficient, 0=ok, 1=warning, 2=critical",
            registry=self.registry,
        )
        self.active_alerts = Gauge(
            "drift_active_alerts",
            "Number of features with critical drift status reported by Core",
            registry=self.registry,
        )
        self.window_size = Gauge(
            "drift_window_size",
            "Configured number of accepted events in one analysis window",
            registry=self.registry,
        )
        self.current_window_events = Gauge(
            "drift_current_window_events",
            "Number of accepted events currently collected in the window",
            registry=self.registry,
        )
        self.events_processed = Counter(
            "drift_events_processed",
            "Successfully accepted Kafka events",
            registry=self.registry,
        )
        self.analysis_runs = Counter(
            "drift_analysis_runs",
            "Successfully completed full-window drift analyses",
            registry=self.registry,
        )
        self.last_analysis_age_seconds = Gauge(
            "drift_last_analysis_age_seconds",
            "Seconds since the latest completed drift analysis",
            registry=self.registry,
        )
        self.report_timestamp = Gauge(
            "drift_report_timestamp_seconds",
            "drift_report.timestamp converted from ISO-8601 UTC to Unix seconds",
            registry=self.registry,
        )

        self.status_feature = Gauge(
            "drift_status_feature",
            "Feature-level drift status",
            ["feature", "type"],
            registry=self.registry,
        )
        self.metric_value = Gauge(
            "drift_metric_value",
            "Drift metric value",
            ["feature", "type", "metric"],
            registry=self.registry,
        )
        self.metric_status = Gauge(
            "drift_status",
            "Metric-level drift status",
            ["feature", "type", "metric"],
            registry=self.registry,
        )
        self.threshold = Gauge(
            "drift_threshold",
            "Configured warning and critical drift thresholds",
            ["metric", "level"],
            registry=self.registry,
        )
        self.resolved_threshold = Gauge(
            "drift_resolved_threshold",
            "Resolved feature-level threshold after local overrides",
            ["feature", "type", "metric", "level"],
            registry=self.registry,
        )
        self.overall_status_streak = Gauge(
            "drift_overall_status_streak",
            "Consecutive analysis windows with the same overall warning/critical status",
            ["status"],
            registry=self.registry,
        )
        self.status_feature_streak = Gauge(
            "drift_status_feature_streak",
            "Consecutive analysis windows with the same feature warning/critical status",
            ["feature", "type", "status"],
            registry=self.registry,
        )
        self.metric_status_streak = Gauge(
            "drift_status_streak",
            "Consecutive analysis windows with the same metric warning/critical status",
            ["feature", "type", "metric", "status"],
            registry=self.registry,
        )

        self.stream_status = Gauge(
            "drift_stream_status",
            "Realtime stream status",
            registry=self.registry,
        )
        self.event_time_lag_seconds = Gauge(
            "drift_event_time_lag_seconds",
            "Processing time minus event_time",
            registry=self.registry,
        )
        self.window_time_span_seconds = Gauge(
            "drift_window_time_span_seconds",
            "Time span covered by the currently collected window",
            registry=self.registry,
        )
        self.max_event_gap_seconds = Gauge(
            "drift_max_event_gap_seconds",
            "Largest event-time gap in the currently collected window",
            registry=self.registry,
        )
        self.invalid_event_time_rate = Gauge(
            "drift_invalid_event_time_rate",
            "Share of records with invalid event_time since the last analysis",
            registry=self.registry,
        )
        self.late_event_rate = Gauge(
            "drift_late_event_rate",
            "Share of late events in the current analysis window",
            registry=self.registry,
        )
        self.late_events = Counter(
            "drift_late_events",
            "Late events since process start",
            registry=self.registry,
        )
        self.out_of_order_events = Counter(
            "drift_out_of_order_events",
            "Out-of-order events since process start",
            registry=self.registry,
        )
        self.stream_threshold = Gauge(
            "drift_stream_threshold",
            "Configured warning and critical stream-quality thresholds",
            ["metric", "level"],
            registry=self.registry,
        )

        self.reference_profile_info = Info(
            "drift_reference_profile",
            "Metadata of the active reference sample",
            registry=self.registry,
        )
        self.reference_sample_size = Gauge(
            "drift_reference_sample_size",
            "Reference sample size used for comparison",
            registry=self.registry,
        )

        self.av_status = Gauge(
            "drift_av_status",
            "Adversarial validation status: -1=not configured, 0=ok, 1=warning, 2=critical",
            registry=self.registry,
        )
        self.av_status_streak = Gauge(
            "drift_av_status_streak",
            "Consecutive analysis windows with the same AV warning/critical status",
            ["status"],
            registry=self.registry,
        )
        self.av_available = Gauge(
            "drift_av_available",
            "Whether an adversarial validation result is available: 0=no, 1=yes",
            registry=self.registry,
        )
        self.av_roc_auc = Gauge(
            "drift_av_roc_auc",
            "Adversarial validation ROC AUC",
            registry=self.registry,
        )
        self.av_roc_auc_cv_std = Gauge(
            "drift_av_roc_auc_cv_std",
            "Standard deviation of fold ROC AUC values in adversarial validation",
            registry=self.registry,
        )
        self.av_roc_auc_cv_mean = Gauge(
            "drift_av_roc_auc_cv_mean",
            "Mean ROC AUC across adversarial-validation CV folds",
            registry=self.registry,
        )
        self.av_roc_auc_cv_min = Gauge(
            "drift_av_roc_auc_cv_min",
            "Worst ROC AUC across adversarial-validation CV folds",
            registry=self.registry,
        )
        self.av_roc_auc_cv_max = Gauge(
            "drift_av_roc_auc_cv_max",
            "Best ROC AUC across adversarial-validation CV folds",
            registry=self.registry,
        )
        self.av_driver_consistency = Gauge(
            "drift_av_driver_consistency",
            "Mean pairwise cosine similarity of feature-importance vectors across CV folds",
            registry=self.registry,
        )
        self.av_driver_similarity_previous = Gauge(
            "drift_av_driver_similarity_previous",
            "Cosine similarity of AV feature importance to the previous completed AV run",
            registry=self.registry,
        )
        self.av_top1_importance_share = Gauge(
            "drift_av_top1_importance_share",
            "Share of total AV feature importance explained by the strongest driver",
            registry=self.registry,
        )
        self.av_top3_importance_share = Gauge(
            "drift_av_top3_importance_share",
            "Share of total AV feature importance explained by the top three drivers",
            registry=self.registry,
        )
        self.av_timestamp = Gauge(
            "drift_av_timestamp_seconds",
            "Timestamp of the latest adversarial validation run",
            registry=self.registry,
        )
        self.av_last_run_timestamp = Gauge(
            "drift_av_last_run_timestamp_seconds",
            "Unix timestamp of the latest adversarial validation run",
            registry=self.registry,
        )
        self.av_dataset_size = Gauge(
            "drift_av_dataset_size",
            "Balanced rows per dataset used by adversarial validation",
            registry=self.registry,
        )
        self.av_reference_rows = Gauge(
            "drift_av_reference_rows",
            "Reference rows available to adversarial validation",
            registry=self.registry,
        )
        self.av_current_rows = Gauge(
            "drift_av_current_rows",
            "Current rows available to adversarial validation",
            registry=self.registry,
        )
        self.av_features_evaluated = Gauge(
            "drift_av_features_evaluated",
            "Number of features evaluated by adversarial validation",
            registry=self.registry,
        )
        self.av_sample_fraction = Gauge(
            "drift_av_sample_fraction",
            "Fraction of each source dataset used in the balanced AV sample",
            ["dataset"],
            registry=self.registry,
        )
        self.av_feature_importance = Gauge(
            "drift_av_feature_importance",
            "Top adversarial-validation feature importance",
            ["feature", "rank"],
            registry=self.registry,
        )
        self.av_threshold = Gauge(
            "drift_av_threshold",
            "Configured adversarial-validation ROC-AUC thresholds",
            ["level"],
            registry=self.registry,
        )

        self.overall_status.set(-1)
        self.active_alerts.set(0)
        self.window_size.set(0)
        self.current_window_events.set(0)
        self.stream_status.set(-1)
        self.last_analysis_age_seconds.set(-1)
        self.report_timestamp.set(0)
        self.av_status.set(-1)
        self.av_available.set(0)
        self.av_roc_auc.set(-1)
        self.av_roc_auc_cv_std.set(-1)
        self.av_roc_auc_cv_mean.set(-1)
        self.av_roc_auc_cv_min.set(-1)
        self.av_roc_auc_cv_max.set(-1)
        self.av_driver_consistency.set(-1)
        self.av_driver_similarity_previous.set(-1)
        self.av_top1_importance_share.set(-1)
        self.av_top3_importance_share.set(-1)
        self.av_timestamp.set(-1)
        self.av_last_run_timestamp.set(-1)
        self.av_dataset_size.set(0)
        self.av_reference_rows.set(0)
        self.av_current_rows.set(0)
        self.av_features_evaluated.set(0)
        self.av_sample_fraction.clear()
        for status_name in ("warning", "critical"):
            self.overall_status_streak.labels(status=status_name).set(0)
            self.av_status_streak.labels(status=status_name).set(0)

        self._last_analysis_monotonic: float | None = None
        self._last_late_events = 0
        self._last_out_of_order_events = 0
        self._overall_streak_state: tuple[int, int] = (-1, 0)
        self._feature_streak_state: dict[tuple[str, str], tuple[int, int]] = {}
        self._metric_streak_state: dict[tuple[str, str, str], tuple[int, int]] = {}
        self._av_streak_state: tuple[int, int] = (-1, 0)
        self._previous_av_importance: dict[str, float] | None = None

    def start_http_server(self, port: int) -> tuple[Any, Any]:
        """Запускает HTTP endpoint Prometheus для текущего реестра."""
        return start_http_server(port, registry=self.registry)

    def set_window_size(self, window_size: int) -> None:
        """Публикует настроенный размер окна анализа."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size.set(window_size)

    def set_current_window_events(self, count: int) -> None:
        """Публикует фактическое число событий в текущем окне."""
        if count < 0:
            raise ValueError("current window event count must be non-negative")
        self.current_window_events.set(count)

    def record_processed_event(self) -> None:
        """Увеличивает счётчик принятых Kafka-событий."""
        self.events_processed.inc()
        self.refresh_analysis_age()

    def record_analysis_run(self) -> None:
        """Фиксирует успешно завершённый анализ полного окна."""
        self.analysis_runs.inc()
        self._last_analysis_monotonic = time.monotonic()
        self.last_analysis_age_seconds.set(0)

    def refresh_analysis_age(self) -> None:
        """Обновляет возраст последнего успешно завершённого анализа."""
        if self._last_analysis_monotonic is not None:
            self.last_analysis_age_seconds.set(
                time.monotonic() - self._last_analysis_monotonic
            )

    def update_reference_profile(self, metadata: dict[str, Any] | None) -> None:
        """Экспортирует метаданные активной reference-выборки."""
        if not metadata:
            return

        self.reference_profile_info.info(
            {
                "dataset_name": str(metadata.get("dataset_name", "")),
                "profile_created_at": str(metadata.get("profile_created_at", "")),
            }
        )
        sample_size = _finite_float(metadata.get("sample_size"))
        if sample_size is not None:
            self.reference_sample_size.set(sample_size)

    def update_drift_thresholds(self, config: Any) -> None:
        """Публикует глобальные Core thresholds, не затирая local override."""
        self.threshold.clear()
        raw_thresholds = getattr(config, "thresholds", None)
        if isinstance(raw_thresholds, dict):
            for metric, pair in raw_thresholds.items():
                metric_name = getattr(metric, "value", str(metric))
                for level in ("warning", "critical"):
                    value = _finite_float(getattr(pair, level, None))
                    if value is not None:
                        self.threshold.labels(metric=metric_name, level=level).set(value)

        prediction = getattr(config, "prediction_metrics", None)
        resolved = getattr(prediction, "resolved_thresholds", None)
        if getattr(prediction, "enabled", False) and isinstance(resolved, dict):
            for metric, pair in resolved.items():
                raw_name = getattr(metric, "value", str(metric))
                metric_name = PREDICTION_METRIC_ALIASES.get(raw_name, raw_name)
                for level in ("warning", "critical"):
                    value = _finite_float(getattr(pair, level, None))
                    if value is not None:
                        self.threshold.labels(metric=metric_name, level=level).set(value)

    def update_stream_thresholds(self, thresholds: Any) -> None:
        """Обновляет stream-quality thresholds в Prometheus."""
        self.stream_threshold.clear()
        as_dict = getattr(thresholds, "as_dict", None)
        if not callable(as_dict):
            return
        for metric, pair in as_dict().items():
            self.stream_threshold.labels(metric=metric, level="warning").set(
                float(pair.warning)
            )
            self.stream_threshold.labels(metric=metric, level="critical").set(
                float(pair.critical)
            )

    def update_stream(self, snapshot: StreamSnapshot) -> None:
        """Экспортирует текущий снимок технического состояния потока."""
        self.stream_status.set(snapshot.status)
        self.event_time_lag_seconds.set(snapshot.event_time_lag_seconds)
        self.window_time_span_seconds.set(snapshot.window_time_span_seconds)
        self.max_event_gap_seconds.set(snapshot.max_event_gap_seconds)
        self.invalid_event_time_rate.set(snapshot.invalid_event_time_rate)
        self.late_event_rate.set(snapshot.late_event_rate)

        late_delta = snapshot.late_events_total - self._last_late_events
        if late_delta > 0:
            self.late_events.inc(late_delta)
        self._last_late_events = snapshot.late_events_total

        out_of_order_delta = (
            snapshot.out_of_order_events_total - self._last_out_of_order_events
        )
        if out_of_order_delta > 0:
            self.out_of_order_events.inc(out_of_order_delta)
        self._last_out_of_order_events = snapshot.out_of_order_events_total

    def update_report(self, report: dict[str, Any]) -> None:
        """Экспортирует Core drift-report без повторного расчёта статусов."""
        self._clear_dynamic_report_series()

        overall_status = _status_number(report.get("overall_status"), -1)
        self.overall_status.set(overall_status)
        self.active_alerts.set(_finite_float(report.get("active_alerts")) or 0)
        self._update_overall_streak(overall_status)

        report_window_size = _finite_float(report.get("window_size"))
        if report_window_size is not None and report_window_size > 0:
            self.window_size.set(report_window_size)

        timestamp = _parse_timestamp_seconds(report.get("timestamp"))
        if timestamp is not None:
            self.report_timestamp.set(timestamp)

        # Legacy/mock report может содержать thresholds на верхнем уровне.
        top_level_thresholds = report.get("thresholds")
        if isinstance(top_level_thresholds, dict):
            for metric, pair in top_level_thresholds.items():
                for level, value in self._parse_threshold_pair(pair).items():
                    self.threshold.labels(metric=str(metric), level=level).set(value)

        present_features: set[tuple[str, str]] = set()
        present_metrics: set[tuple[str, str, str]] = set()

        features = report.get("features")
        if isinstance(features, dict):
            for feature, block in features.items():
                if isinstance(block, dict):
                    feature_key, metric_keys = self._export_feature(
                        feature=str(feature),
                        block=block,
                        prediction=False,
                    )
                    present_features.add(feature_key)
                    present_metrics.update(metric_keys)

        prediction = report.get("prediction")
        if isinstance(prediction, dict):
            feature_key, metric_keys = self._export_feature(
                feature="prediction",
                block=prediction,
                prediction=True,
            )
            present_features.add(feature_key)
            present_metrics.update(metric_keys)

        self._drop_missing_streak_state(present_features, present_metrics)

        if "adversarial_validation" in report:
            self.update_adversarial_validation(report.get("adversarial_validation"))

    def update_adversarial_validation(self, block: Any) -> None:
        """Экспортирует optional monitoring-блок, полученный из Core AV."""
        self.av_feature_importance.clear()
        self.av_threshold.clear()
        self.av_status.set(-1)
        self.av_available.set(0)
        self.av_roc_auc.set(-1)
        self.av_roc_auc_cv_std.set(-1)
        self.av_roc_auc_cv_mean.set(-1)
        self.av_roc_auc_cv_min.set(-1)
        self.av_roc_auc_cv_max.set(-1)
        self.av_driver_consistency.set(-1)
        self.av_driver_similarity_previous.set(-1)
        self.av_top1_importance_share.set(-1)
        self.av_top3_importance_share.set(-1)
        self.av_timestamp.set(-1)
        self.av_last_run_timestamp.set(-1)
        self.av_dataset_size.set(0)
        self.av_reference_rows.set(0)
        self.av_current_rows.set(0)
        self.av_features_evaluated.set(0)
        self.av_sample_fraction.clear()

        if not isinstance(block, dict):
            self._update_av_streak(-1)
            return

        self.av_available.set(1)
        av_status = _status_number(block.get("status"), -1)
        self.av_status.set(av_status)
        self._update_av_streak(av_status)

        for key, gauge in (
            ("roc_auc", self.av_roc_auc),
            ("roc_auc_cv_std", self.av_roc_auc_cv_std),
            ("roc_auc_cv_mean", self.av_roc_auc_cv_mean),
            ("roc_auc_cv_min", self.av_roc_auc_cv_min),
            ("roc_auc_cv_max", self.av_roc_auc_cv_max),
            ("driver_consistency", self.av_driver_consistency),
            ("driver_similarity_previous", self.av_driver_similarity_previous),
            ("top1_importance_share", self.av_top1_importance_share),
            ("top3_importance_share", self.av_top3_importance_share),
        ):
            value = _finite_float(block.get(key))
            if value is not None:
                gauge.set(value)

        timestamp = _parse_timestamp_seconds(block.get("timestamp"))
        if timestamp is not None:
            self.av_timestamp.set(timestamp)
            self.av_last_run_timestamp.set(timestamp)

        reference_rows = _finite_float(block.get("reference_rows"))
        current_rows = _finite_float(block.get("current_rows"))
        if reference_rows is not None:
            self.av_reference_rows.set(reference_rows)
        if current_rows is not None:
            self.av_current_rows.set(current_rows)

        dataset_size = _finite_float(block.get("dataset_size"))
        if dataset_size is None and reference_rows is not None and current_rows is not None:
            dataset_size = min(reference_rows, current_rows)
        if dataset_size is not None:
            self.av_dataset_size.set(dataset_size)

        features_evaluated = _finite_float(block.get("features_evaluated"))
        if features_evaluated is not None:
            self.av_features_evaluated.set(features_evaluated)

        reference_fraction = _finite_float(block.get("reference_sample_fraction"))
        if (
            reference_fraction is None
            and dataset_size is not None
            and reference_rows is not None
            and reference_rows > 0
        ):
            reference_fraction = dataset_size / reference_rows
        if reference_fraction is not None:
            self.av_sample_fraction.labels(dataset="reference").set(reference_fraction)

        current_fraction = _finite_float(block.get("current_sample_fraction"))
        if (
            current_fraction is None
            and dataset_size is not None
            and current_rows is not None
            and current_rows > 0
        ):
            current_fraction = dataset_size / current_rows
        if current_fraction is not None:
            self.av_sample_fraction.labels(dataset="current").set(current_fraction)

        raw_thresholds = block.get("thresholds")
        if isinstance(raw_thresholds, dict):
            for level in ("warning", "critical"):
                value = _finite_float(raw_thresholds.get(level))
                if value is not None:
                    self.av_threshold.labels(level=level).set(value)

        raw_importance = block.get("feature_importance")
        if not isinstance(raw_importance, list):
            return
        for item in raw_importance:
            if not isinstance(item, dict):
                continue
            feature = item.get("feature")
            importance = _finite_float(item.get("importance"))
            rank = _finite_float(item.get("rank"))
            if feature is None or importance is None or rank is None:
                continue
            self.av_feature_importance.labels(
                feature=str(feature),
                rank=str(int(rank)),
            ).set(importance)

    def update_adversarial_validation_result(
        self,
        roc_auc: float,
        importance: Any,
        *,
        timestamp: Any,
        thresholds: Any,
        reference_rows: int,
        current_rows: int,
        top_features: int = 10,
    ) -> None:
        """Преобразует публичный результат Core AV в monitoring-метрики.

        Core API возвращает общий ROC AUC и feature importance. Fold-level AUC
        экспортируются только при наличии соответствующих значений в
        ``importance.attrs``; realtime-слой не вычисляет их повторно.
        """
        warning = _finite_float(getattr(thresholds, "warning", None))
        critical = _finite_float(getattr(thresholds, "critical", None))
        status = -1
        if warning is not None and critical is not None:
            if roc_auc >= critical:
                status = 2
            elif roc_auc >= warning:
                status = 1
            else:
                status = 0

        rows: list[dict[str, Any]] = []
        if hasattr(importance, "head") and hasattr(importance, "to_dict"):
            required = {"feature", "importance", "rank"}
            columns = set(getattr(importance, "columns", []))
            if required.issubset(columns):
                rows = (
                    importance.head(top_features)
                    .loc[:, ["feature", "importance", "rank"]]
                    .to_dict(orient="records")
                )

        current_importance: dict[str, float] = {}
        if hasattr(importance, "iterrows"):
            for _, row in importance.iterrows():
                feature = row.get("feature")
                value = _finite_float(row.get("importance"))
                if feature is not None and value is not None:
                    current_importance[str(feature)] = value

        driver_similarity_previous = None
        if self._previous_av_importance and current_importance:
            names = set(self._previous_av_importance) | set(current_importance)
            dot = sum(
                self._previous_av_importance.get(name, 0.0)
                * current_importance.get(name, 0.0)
                for name in names
            )
            previous_norm = math.sqrt(
                sum(self._previous_av_importance.get(name, 0.0) ** 2 for name in names)
            )
            current_norm = math.sqrt(
                sum(current_importance.get(name, 0.0) ** 2 for name in names)
            )
            if previous_norm and current_norm:
                driver_similarity_previous = dot / (previous_norm * current_norm)
        self._previous_av_importance = current_importance

        top1 = sum(
            current_importance.get(str(row.get("feature")), 0.0) for row in rows[:1]
        )
        top3 = sum(
            current_importance.get(str(row.get("feature")), 0.0) for row in rows[:3]
        )
        dataset_size = min(reference_rows, current_rows)

        block: dict[str, Any] = {
            "timestamp": timestamp,
            "status": status,
            "roc_auc": roc_auc,
            "reference_rows": reference_rows,
            "current_rows": current_rows,
            "dataset_size": dataset_size,
            "features_evaluated": len(current_importance),
            "reference_sample_fraction": (
                dataset_size / reference_rows if reference_rows else 0.0
            ),
            "current_sample_fraction": (
                dataset_size / current_rows if current_rows else 0.0
            ),
            "top1_importance_share": top1,
            "top3_importance_share": top3,
            "feature_importance": rows,
        }
        if warning is not None and critical is not None:
            block["thresholds"] = {"warning": warning, "critical": critical}

        attrs = getattr(importance, "attrs", {})
        if isinstance(attrs, dict):
            for name in (
                "roc_auc_cv_mean",
                "roc_auc_cv_std",
                "roc_auc_cv_min",
                "roc_auc_cv_max",
                "driver_consistency",
            ):
                value = _finite_float(attrs.get(name))
                if value is not None:
                    block[name] = value
        if driver_similarity_previous is not None:
            block["driver_similarity_previous"] = driver_similarity_previous

        self.update_adversarial_validation(block)

    def _clear_dynamic_report_series(self) -> None:
        """Очищает label-серии, полностью заменяемые новым Core-report."""
        self.status_feature.clear()
        self.metric_value.clear()
        self.metric_status.clear()
        self.resolved_threshold.clear()
        self.status_feature_streak.clear()
        self.metric_status_streak.clear()

    @staticmethod
    def _parse_threshold_pair(pair: Any) -> dict[str, float]:
        """Извлекает конечные warning/critical значения из metric payload."""
        if not isinstance(pair, dict):
            return {}
        result: dict[str, float] = {}
        for level in ("warning", "critical"):
            value = _finite_float(pair.get(level))
            if value is not None:
                result[level] = value
        return result

    @staticmethod
    def _next_streak(
        previous: tuple[int, int] | None,
        current_status: int,
    ) -> tuple[int, int]:
        """Обновляет длину серии одинаковых warning/critical состояний."""
        if current_status not in (1, 2):
            return current_status, 0
        if previous is not None and previous[0] == current_status:
            return current_status, previous[1] + 1
        return current_status, 1

    def _set_streak_labels(
        self,
        gauge: Gauge,
        labels: dict[str, str],
        state: tuple[int, int],
    ) -> None:
        """Публикует warning/critical streak в отдельные label-серии."""
        current_status, count = state
        for status_number, status_name in ((1, "warning"), (2, "critical")):
            gauge.labels(**labels, status=status_name).set(
                count if current_status == status_number else 0
            )

    def _update_overall_streak(self, current_status: int) -> None:
        """Обновляет streak общего drift status."""
        self._overall_streak_state = self._next_streak(
            self._overall_streak_state,
            current_status,
        )
        self._set_streak_labels(
            self.overall_status_streak,
            {},
            self._overall_streak_state,
        )

    def _update_av_streak(self, current_status: int) -> None:
        """Обновляет streak adversarial-validation status."""
        self._av_streak_state = self._next_streak(
            self._av_streak_state,
            current_status,
        )
        self._set_streak_labels(
            self.av_status_streak,
            {},
            self._av_streak_state,
        )

    def _update_feature_streak(
        self,
        key: tuple[str, str],
        current_status: int,
    ) -> None:
        """Обновляет streak для одной feature."""
        state = self._next_streak(self._feature_streak_state.get(key), current_status)
        self._feature_streak_state[key] = state
        self._set_streak_labels(
            self.status_feature_streak,
            {"feature": key[0], "type": key[1]},
            state,
        )

    def _update_metric_streak(
        self,
        key: tuple[str, str, str],
        current_status: int,
    ) -> None:
        """Обновляет streak для одной feature metric."""
        state = self._next_streak(self._metric_streak_state.get(key), current_status)
        self._metric_streak_state[key] = state
        self._set_streak_labels(
            self.metric_status_streak,
            {"feature": key[0], "type": key[1], "metric": key[2]},
            state,
        )

    def _drop_missing_streak_state(
        self,
        present_features: set[tuple[str, str]],
        present_metrics: set[tuple[str, str, str]],
    ) -> None:
        """Удаляет streak state для label-серий, исчезнувших из нового report."""
        self._feature_streak_state = {
            key: value
            for key, value in self._feature_streak_state.items()
            if key in present_features
        }
        self._metric_streak_state = {
            key: value
            for key, value in self._metric_streak_state.items()
            if key in present_metrics
        }

    def _export_feature(
        self,
        *,
        feature: str,
        block: dict[str, Any],
        prediction: bool,
    ) -> tuple[tuple[str, str], set[tuple[str, str, str]]]:
        """Экспортирует feature/prediction block без пересчёта Core status."""
        feature_type = str(
            block.get("type") or ("numeric" if prediction else "unknown")
        )
        raw_alerts = block.get("alerts")
        alerts = (
            [str(item) for item in raw_alerts]
            if isinstance(raw_alerts, list)
            else []
        )

        metrics = block.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        metric_status_numbers: list[int] = []
        metric_keys: set[tuple[str, str, str]] = set()
        for raw_metric, payload in metrics.items():
            raw_metric = str(raw_metric)
            metric = (
                PREDICTION_METRIC_ALIASES.get(raw_metric, raw_metric)
                if prediction
                else raw_metric
            )

            value: float | None
            metric_status: str | int | None = None
            resolved_thresholds: dict[str, float] = {}
            if isinstance(payload, dict):
                value = _finite_float(payload.get("value"))
                metric_status = payload.get("status")
                resolved_thresholds = self._parse_threshold_pair(payload)
            else:
                value = _finite_float(payload)

            if value is None:
                continue
            if metric_status is None:
                metric_status = _alert_status(raw_metric, alerts)
            status_number = _status_number(metric_status, 0)
            metric_status_numbers.append(status_number)

            metric_labels = {
                "feature": feature,
                "type": feature_type,
                "metric": metric,
            }
            self.metric_value.labels(**metric_labels).set(value)
            self.metric_status.labels(**metric_labels).set(status_number)
            for level, threshold_value in resolved_thresholds.items():
                self.resolved_threshold.labels(
                    **metric_labels,
                    level=level,
                ).set(threshold_value)

            metric_key = (feature, feature_type, metric)
            metric_keys.add(metric_key)
            self._update_metric_streak(metric_key, status_number)

        declared_status = block.get("status")
        feature_status = (
            max(metric_status_numbers, default=0)
            if declared_status is None
            else _status_number(declared_status, 0)
        )
        self.status_feature.labels(
            feature=feature,
            type=feature_type,
        ).set(feature_status)
        feature_key = (feature, feature_type)
        self._update_feature_streak(feature_key, feature_status)
        return feature_key, metric_keys
