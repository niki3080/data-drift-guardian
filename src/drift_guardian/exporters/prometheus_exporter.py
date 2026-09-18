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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if value is None:
        return default
    return STATUS_TO_NUMBER.get(str(value).lower(), default)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_timestamp_seconds(value: Any) -> float | None:
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
            "Number of active drift alerts reported by the analyzer",
            registry=self.registry,
        )
        self.window_size = Gauge(
            "drift_window_size",
            "Configured number of accepted events in one analysis window",
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

        self.overall_status.set(-1)
        self.active_alerts.set(0)
        self.window_size.set(0)
        self.stream_status.set(-1)
        self.last_analysis_age_seconds.set(-1)
        self.report_timestamp.set(0)

        self._last_analysis_monotonic: float | None = None
        self._last_late_events = 0
        self._last_out_of_order_events = 0

    def start_http_server(self, port: int) -> tuple[Any, Any]:
        """Запускает HTTP endpoint Prometheus для текущего реестра."""
        return start_http_server(port, registry=self.registry)

    def set_window_size(self, window_size: int) -> None:
        """Публикует настроенный размер окна анализа."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size.set(window_size)

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
        """Преобразует drift-report в согласованный контракт Prometheus."""
        self._clear_dynamic_report_series()

        self.overall_status.set(_status_number(report.get("overall_status"), -1))
        self.active_alerts.set(_finite_float(report.get("active_alerts")) or 0)

        report_window_size = _finite_float(report.get("window_size"))
        if report_window_size is not None and report_window_size > 0:
            self.window_size.set(report_window_size)

        timestamp = _parse_timestamp_seconds(report.get("timestamp"))
        if timestamp is not None:
            self.report_timestamp.set(timestamp)

        thresholds: dict[str, dict[str, float]] = {}
        top_level_thresholds = report.get("thresholds")
        if isinstance(top_level_thresholds, dict):
            for metric, pair in top_level_thresholds.items():
                parsed = self._parse_threshold_pair(pair)
                if parsed:
                    thresholds[str(metric)] = parsed

        features = report.get("features")
        if isinstance(features, dict):
            for feature, block in features.items():
                if isinstance(block, dict):
                    self._export_feature(
                        feature=str(feature),
                        block=block,
                        thresholds=thresholds,
                        prediction=False,
                    )

        prediction = report.get("prediction")
        if isinstance(prediction, dict):
            self._export_feature(
                feature="prediction",
                block=prediction,
                thresholds=thresholds,
                prediction=True,
            )

        for metric, pair in thresholds.items():
            for level in ("warning", "critical"):
                value = pair.get(level)
                if value is not None:
                    self.threshold.labels(metric=metric, level=level).set(value)

    def _clear_dynamic_report_series(self) -> None:
        self.status_feature.clear()
        self.metric_value.clear()
        self.metric_status.clear()
        self.threshold.clear()

    @staticmethod
    def _parse_threshold_pair(pair: Any) -> dict[str, float]:
        if not isinstance(pair, dict):
            return {}
        result: dict[str, float] = {}
        for level in ("warning", "critical"):
            value = _finite_float(pair.get(level))
            if value is not None:
                result[level] = value
        return result

    def _export_feature(
        self,
        *,
        feature: str,
        block: dict[str, Any],
        thresholds: dict[str, dict[str, float]],
        prediction: bool,
    ) -> None:
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

        for raw_metric, payload in metrics.items():
            raw_metric = str(raw_metric)
            metric = (
                PREDICTION_METRIC_ALIASES.get(raw_metric, raw_metric)
                if prediction
                else raw_metric
            )
            if prediction and metric == "prediction_psi":
                source = thresholds.get(raw_metric) or thresholds.get("psi")
                if source and metric not in thresholds:
                    thresholds[metric] = dict(source)

            value: float | None
            metric_status: str | int | None = None
            nested_thresholds: dict[str, float] = {}

            if isinstance(payload, dict):
                value = _finite_float(payload.get("value"))
                metric_status = payload.get("status")
                nested_thresholds = self._parse_threshold_pair(payload)
            else:
                value = _finite_float(payload)

            if value is None:
                continue

            if nested_thresholds:
                thresholds.setdefault(metric, {}).update(nested_thresholds)

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

        declared_status = block.get("status")
        if declared_status is None:
            feature_status = max(metric_status_numbers, default=0)
        else:
            feature_status = _status_number(declared_status, 0)

        self.status_feature.labels(
            feature=feature,
            type=feature_type,
        ).set(feature_status)
