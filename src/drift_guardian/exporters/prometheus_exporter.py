from __future__ import annotations

from datetime import datetime
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    disable_created_metrics,
    start_http_server,
)

from drift_guardian.ingestion.stream_metrics import StreamSnapshot

STATUS_TO_NUMBER = {
    "insufficient_data": -1,
    "ok": 0,
    "warning": 1,
    "critical": 2,
}


disable_created_metrics()


class PrometheusExporter:
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
        self.report_timestamp = Gauge(
            "drift_report_timestamp_seconds",
            "Timestamp of the latest drift report",
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

        self.metric_value = Gauge(
            "drift_metric_value",
            "Drift metric value",
            ["feature", "metric"],
            registry=self.registry,
        )
        self.threshold = Gauge(
            "drift_threshold",
            "Drift metric threshold",
            ["feature", "metric", "level"],
            registry=self.registry,
        )
        self.metric_status = Gauge(
            "drift_status",
            "Drift metric status",
            ["feature", "metric"],
            registry=self.registry,
        )
        self.feature_status = Gauge(
            "drift_status_feature",
            "Feature drift status",
            ["feature"],
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
            "Share of observed records with invalid event_time",
            registry=self.registry,
        )
        self.late_events = Counter(
            "drift_late_events",
            "Late events",
            registry=self.registry,
        )
        self.out_of_order_events = Counter(
            "drift_out_of_order_events",
            "Out-of-order events",
            registry=self.registry,
        )

        self.overall_status.set(-1)
        self.stream_status.set(-1)
        self._last_late_events = 0
        self._last_out_of_order_events = 0

    def start_http_server(self, port: int) -> tuple[Any, Any]:
        return start_http_server(port, registry=self.registry)

    def record_processed_event(self) -> None:
        self.events_processed.inc()

    def update_stream(self, snapshot: StreamSnapshot) -> None:
        self.stream_status.set(snapshot.status)
        self.window_size.set(snapshot.window_size)
        self.event_time_lag_seconds.set(snapshot.event_time_lag_seconds)
        self.window_time_span_seconds.set(snapshot.window_time_span_seconds)
        self.max_event_gap_seconds.set(snapshot.max_event_gap_seconds)
        self.invalid_event_time_rate.set(snapshot.invalid_event_time_rate)

        late_events = snapshot.late_events_total - self._last_late_events
        if late_events > 0:
            self.late_events.inc(late_events)
        self._last_late_events = snapshot.late_events_total

        out_of_order_events = (
            snapshot.out_of_order_events_total
            - self._last_out_of_order_events
        )
        if out_of_order_events > 0:
            self.out_of_order_events.inc(out_of_order_events)
        self._last_out_of_order_events = snapshot.out_of_order_events_total

    def update_report(self, report: dict[str, Any]) -> None:
        self.overall_status.set(self._status(report.get("overall_status")))
        self.active_alerts.set(float(report.get("active_alerts", 0)))
        self.window_size.set(float(report.get("window_size", 0)))
        self._set_report_timestamp(report.get("timestamp"))

        for feature, payload in (report.get("features") or {}).items():
            feature_name = str(feature)
            self.feature_status.labels(feature=feature_name).set(
                self._status(payload.get("status"))
            )
            self._export_metrics(feature_name, payload.get("metrics") or {})

        prediction = report.get("prediction")
        if prediction:
            self.feature_status.labels(feature="prediction").set(
                self._status(prediction.get("status"))
            )
            self._export_metrics(
                "prediction",
                prediction.get("metrics") or {},
            )

    def _export_metrics(
        self,
        feature: str,
        metrics: dict[str, Any],
    ) -> None:
        for metric_name, payload in metrics.items():
            if not isinstance(payload, dict):
                continue

            value = payload.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            labels = {"feature": feature, "metric": str(metric_name)}
            self.metric_value.labels(**labels).set(float(value))

            for level in ("warning", "critical"):
                threshold = payload.get(level)
                if isinstance(threshold, (int, float)) and not isinstance(
                    threshold,
                    bool,
                ):
                    self.threshold.labels(
                        feature=feature,
                        metric=str(metric_name),
                        level=level,
                    ).set(float(threshold))

            status = payload.get("status")
            if status is not None:
                self.metric_status.labels(**labels).set(self._status(status))

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
