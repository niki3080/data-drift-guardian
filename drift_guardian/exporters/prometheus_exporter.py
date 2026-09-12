from __future__ import annotations

from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    disable_created_metrics,
    start_http_server,
)

from drift_guardian.realtime.realtime_monitor import StreamSnapshot

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
        self.window_size = Gauge(
            "drift_window_size",
            "Current sliding-window size",
            registry=self.registry,
        )
        self.events_processed = Counter(
            "drift_events_processed_total",
            "Successfully processed Kafka events",
            registry=self.registry,
        )

        labels = ["feature", "type"]
        self.feature_status = Gauge(
            "drift_feature_status",
            "Per-feature drift status",
            labels,
            registry=self.registry,
        )
        self.feature_metrics = {
            "psi": Gauge(
                "drift_feature_psi",
                "Feature PSI",
                labels,
                registry=self.registry,
            ),
            "missing_rate": Gauge(
                "drift_feature_missing_rate",
                "Feature missing rate",
                labels,
                registry=self.registry,
            ),
            "mean_zscore": Gauge(
                "drift_feature_mean_zscore",
                "Feature mean z-score",
                labels,
                registry=self.registry,
            ),
            "unseen_category_rate": Gauge(
                "drift_feature_unseen_category_rate",
                "Feature unseen-category rate",
                labels,
                registry=self.registry,
            ),
            "cardinality_ratio": Gauge(
                "drift_feature_cardinality_ratio",
                "Feature cardinality ratio",
                labels,
                registry=self.registry,
            ),
        }

        self.prediction_status = Gauge(
            "drift_prediction_status",
            "Prediction drift status",
            registry=self.registry,
        )
        self.prediction_psi = Gauge(
            "drift_prediction_psi",
            "Prediction PSI",
            registry=self.registry,
        )
        self.prediction_positive_rate = Gauge(
            "drift_prediction_positive_rate",
            "Prediction positive rate",
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
            "Share of records with invalid event_time",
            registry=self.registry,
        )
        self.late_events = Counter(
            "drift_late_events_total",
            "Late events",
            registry=self.registry,
        )
        self.out_of_order_events = Counter(
            "drift_out_of_order_events_total",
            "Out-of-order events",
            registry=self.registry,
        )

        self.overall_status.set(-1)
        self.prediction_status.set(-1)
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

        for feature, payload in (report.get("features") or {}).items():
            feature_type = str(payload.get("type", "unknown"))
            labels = {
                "feature": str(feature),
                "type": feature_type,
            }
            self.feature_status.labels(**labels).set(
                self._status(payload.get("status"))
            )

            metrics = payload.get("metrics") or {}
            for name, gauge in self.feature_metrics.items():
                value = metrics.get(name)
                if value is not None:
                    gauge.labels(**labels).set(float(value))

        prediction = report.get("prediction")
        if prediction:
            self.prediction_status.set(
                self._status(prediction.get("status"))
            )
            metrics = prediction.get("metrics") or {}

            prediction_psi = metrics.get("prediction_psi")
            if prediction_psi is not None:
                self.prediction_psi.set(float(prediction_psi))

            positive_rate = metrics.get("positive_prediction_rate")
            if positive_rate is not None:
                self.prediction_positive_rate.set(float(positive_rate))

    @staticmethod
    def _status(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)

        return float(STATUS_TO_NUMBER.get(str(value), -1))
