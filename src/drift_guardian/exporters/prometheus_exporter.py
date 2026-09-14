from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Info,
    disable_created_metrics,
    start_http_server,
)

from drift_guardian.ingestion.stream_metrics import StreamSnapshot, StreamThresholds

STATUS_TO_NUMBER = {
    "insufficient_data": -1,
    "ok": 0,
    "passed": 0,
    "warning": 1,
    "critical": 2,
}

FEATURE_METRIC_NAMES = {
    "psi": "drift_feature_psi",
    "missing_rate": "drift_feature_missing_rate",
    # Kept while it is still present in main/Grafana; the team plans to remove it.
    "mean_zscore": "drift_feature_mean_zscore",
    "unseen_category_rate": "drift_feature_unseen_category_rate",
    "cardinality_ratio": "drift_feature_cardinality_ratio",
    "cramer_v_score": "drift_feature_cramer_v_score",
}


disable_created_metrics()


class PrometheusExporter:
    """Exports the realtime pipeline using the metric contract already in main."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.overall_status = Gauge(
            "drift_overall_status",
            "Overall drift status",
            registry=self.registry,
        )
        self.active_alerts = Gauge(
            "drift_active_alerts",
            "Number of active drift alerts",
            registry=self.registry,
        )
        self.window_size = Gauge(
            "drift_window_size",
            "Current sliding-window size",
            registry=self.registry,
        )
        self.events_processed = Counter(
            "drift_events_processed",
            "Successfully processed Kafka events",
            registry=self.registry,
        )
        self.analysis_runs = Counter(
            "drift_analysis_runs",
            "Successfully completed drift analyses",
            registry=self.registry,
        )
        self.last_analysis_age_seconds = Gauge(
            "drift_last_analysis_age_seconds",
            "Seconds since the latest completed analysis",
            registry=self.registry,
        )
        self.report_timestamp = Gauge(
            "drift_report_timestamp_seconds",
            "Timestamp of the latest drift report",
            registry=self.registry,
        )

        feature_labels = ["feature", "type"]
        self.feature_status = Gauge(
            "drift_feature_status",
            "Per-feature drift status",
            feature_labels,
            registry=self.registry,
        )
        self.feature_active_alerts = Gauge(
            "drift_feature_active_alerts",
            "Number of active alerts for a feature",
            feature_labels,
            registry=self.registry,
        )
        self.feature_alert_info = Gauge(
            "drift_feature_alert_info",
            "Active feature alert",
            ["feature", "type", "alert"],
            registry=self.registry,
        )
        self.feature_metrics = {
            metric: Gauge(
                prometheus_name,
                f"Feature {metric}",
                feature_labels,
                registry=self.registry,
            )
            for metric, prometheus_name in FEATURE_METRIC_NAMES.items()
        }
        self.metric_threshold = Gauge(
            "drift_metric_threshold",
            "Configured warning and critical drift thresholds",
            ["metric", "level"],
            registry=self.registry,
        )

        self.prediction_status = Gauge(
            "drift_prediction_status",
            "Prediction drift status",
            registry=self.registry,
        )
        self.prediction_score_drift = Gauge(
            "drift_prediction_score_drift",
            "Prediction score drift metric",
            registry=self.registry,
        )
        self.prediction_positive_rate = Gauge(
            "drift_prediction_positive_rate",
            "Positive prediction rate",
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
            "Time span covered by the current window",
            registry=self.registry,
        )
        self.max_event_gap_seconds = Gauge(
            "drift_max_event_gap_seconds",
            "Largest event-time gap in the current window",
            registry=self.registry,
        )
        self.invalid_event_time_rate = Gauge(
            "drift_invalid_event_time_rate",
            "Share of records with invalid event_time in the quality window",
            registry=self.registry,
        )
        self.late_event_rate = Gauge(
            "drift_late_event_rate",
            "Share of late events in the quality window",
            registry=self.registry,
        )
        self.out_of_order_event_rate = Gauge(
            "drift_out_of_order_event_rate",
            "Share of out-of-order events in the quality window",
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
            "Metadata of the active reference profile",
            registry=self.registry,
        )
        self.reference_sample_size = Gauge(
            "drift_reference_sample_size",
            "Reference sample size used for comparison",
            registry=self.registry,
        )

        self.overall_status.set(-1)
        self.prediction_status.set(-1)
        self.stream_status.set(-1)
        self.last_analysis_age_seconds.set(-1)
        self._last_analysis_monotonic: float | None = None
        self._last_late_events = 0
        self._last_out_of_order_events = 0

    def start_http_server(self, port: int) -> tuple[Any, Any]:
        return start_http_server(port, registry=self.registry)

    def record_processed_event(self) -> None:
        self.events_processed.inc()
        self.refresh_analysis_age()

    def record_analysis_run(self) -> None:
        self.analysis_runs.inc()
        self._last_analysis_monotonic = time.monotonic()
        self.last_analysis_age_seconds.set(0)

    def refresh_analysis_age(self) -> None:
        if self._last_analysis_monotonic is not None:
            self.last_analysis_age_seconds.set(
                time.monotonic() - self._last_analysis_monotonic
            )

    def set_stream_thresholds(self, thresholds: StreamThresholds) -> None:
        for metric, pair in thresholds.as_dict().items():
            self.stream_threshold.labels(metric=metric, level="warning").set(
                pair.warning
            )
            self.stream_threshold.labels(metric=metric, level="critical").set(
                pair.critical
            )

    def update_reference_profile(self, metadata: dict[str, Any] | None) -> None:
        """Publish metadata exposed by the profiler when that hook is available."""
        if not metadata:
            return

        info: dict[str, str] = {}
        for key in (
            "profile_name",
            "profile_version",
            "profile_created_at",
            "dataset_name",
        ):
            value = metadata.get(key)
            if value is not None:
                info[key] = str(value)
        if info:
            self.reference_profile_info.info(info)

        sample_size = metadata.get("sample_size")
        if isinstance(sample_size, (int, float)) and not isinstance(sample_size, bool):
            self.reference_sample_size.set(float(sample_size))

    def update_stream(self, snapshot: StreamSnapshot) -> None:
        self.stream_status.set(snapshot.status)
        self.window_size.set(snapshot.window_size)
        self.event_time_lag_seconds.set(snapshot.event_time_lag_seconds)
        self.window_time_span_seconds.set(snapshot.window_time_span_seconds)
        self.max_event_gap_seconds.set(snapshot.max_event_gap_seconds)
        self.invalid_event_time_rate.set(snapshot.invalid_event_time_rate)
        self.late_event_rate.set(snapshot.late_event_rate)
        self.out_of_order_event_rate.set(snapshot.out_of_order_event_rate)

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
        self.overall_status.set(self._status(report.get("overall_status")))
        self.active_alerts.set(float(report.get("active_alerts", 0)))
        if isinstance(report.get("window_size"), (int, float)):
            self.window_size.set(float(report["window_size"]))
        self._set_report_timestamp(report.get("timestamp"))

        # Support metadata if the report producer chooses to attach profiler meta.
        reference_metadata = report.get("reference_profile")
        if not isinstance(reference_metadata, dict):
            metadata = report.get("metadata")
            if isinstance(metadata, dict) and "sample_size" in metadata:
                reference_metadata = metadata
        if isinstance(reference_metadata, dict):
            self.update_reference_profile(reference_metadata)

        # Thresholds are global per metric, not per feature.
        self._export_thresholds(report.get("thresholds"))

        self.feature_alert_info.clear()
        for feature, payload in (report.get("features") or {}).items():
            if not isinstance(payload, dict):
                continue

            feature_name = str(feature)
            feature_type = str(payload.get("type", "unknown"))
            labels = {"feature": feature_name, "type": feature_type}
            self.feature_status.labels(**labels).set(
                self._status(payload.get("status"))
            )

            metrics = payload.get("metrics") or {}
            self._export_feature_metrics(labels, metrics)

            alerts = payload.get("alerts")
            if isinstance(alerts, list):
                active_alerts = [str(alert) for alert in alerts]
            else:
                active_alerts = self._alerts_from_metric_payloads(metrics)

            self.feature_active_alerts.labels(**labels).set(len(active_alerts))
            for alert in active_alerts:
                self.feature_alert_info.labels(
                    feature=feature_name,
                    type=feature_type,
                    alert=alert,
                ).set(1)

        self._export_prediction(report.get("prediction"))

    def _export_feature_metrics(
        self,
        labels: dict[str, str],
        metrics: dict[str, Any],
    ) -> None:
        for metric_name, payload in metrics.items():
            metric_name = str(metric_name)
            gauge = self.feature_metrics.get(metric_name)
            if gauge is None:
                continue

            value = payload.get("value") if isinstance(payload, dict) else payload
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                gauge.labels(**labels).set(float(value))

            if isinstance(payload, dict):
                self._export_thresholds({metric_name: payload})

    def _export_thresholds(self, thresholds: Any) -> None:
        if not isinstance(thresholds, dict):
            return
        for metric_name, pair in thresholds.items():
            if not isinstance(pair, dict):
                continue
            for level in ("warning", "critical"):
                value = pair.get(level)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.metric_threshold.labels(
                        metric=str(metric_name),
                        level=level,
                    ).set(float(value))

    @staticmethod
    def _alerts_from_metric_payloads(metrics: dict[str, Any]) -> list[str]:
        alerts: list[str] = []
        for metric_name, payload in metrics.items():
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status", ""))
            if status in {"warning", "critical"}:
                alerts.append(f"{metric_name}_{status}")
        return alerts

    def _export_prediction(self, prediction: Any) -> None:
        if not isinstance(prediction, dict):
            return
        self.prediction_status.set(self._status(prediction.get("status")))
        metrics = prediction.get("metrics") or {}
        if not isinstance(metrics, dict):
            return
        self._set_numeric_metric(
            metrics.get("prediction_score_drift"),
            self.prediction_score_drift,
        )
        self._set_numeric_metric(
            metrics.get("positive_prediction_rate"),
            self.prediction_positive_rate,
        )

    @staticmethod
    def _set_numeric_metric(value: Any, gauge: Gauge) -> None:
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            gauge.set(float(value))

    def _set_report_timestamp(self, value: Any) -> None:
        if not isinstance(value, str):
            return
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return
        if timestamp.tzinfo is not None:
            self.report_timestamp.set(timestamp.timestamp())

    @staticmethod
    def _status(value: Any) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return float(STATUS_TO_NUMBER.get(str(value), -1))
