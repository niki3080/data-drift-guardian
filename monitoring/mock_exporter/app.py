import random
import threading
import time

from prometheus_client import Counter, Gauge, start_http_server


EVENT_INTERVAL_SECONDS = 0.5
WINDOW_SIZE = 1000

NORMAL_DURATION_SECONDS = 60
WARNING_DURATION_SECONDS = 45
CRITICAL_DURATION_SECONDS = 30
STATE_CYCLE_SECONDS = (
    NORMAL_DURATION_SECONDS
    + WARNING_DURATION_SECONDS
    + CRITICAL_DURATION_SECONDS
)


# Глобальные метрики
overall_status = Gauge(
    "drift_overall_status",
    "Overall drift status: -1=insufficient data, 0=ok, 1=warning, 2=critical",
)

active_alerts = Gauge(
    "drift_active_alerts",
    "Number of currently active drift alerts",
)

window_size = Gauge(
    "drift_window_size",
    "Configured number of events in each analysis window",
)

analysis_runs = Counter(
    "drift_analysis_runs",
    "Total number of completed drift analysis runs",
)

last_analysis_age_seconds = Gauge(
    "drift_last_analysis_age_seconds",
    "Seconds elapsed since the last completed drift analysis",
)

window_size.set(WINDOW_SIZE)
last_analysis_age_seconds.set(-1)

events_processed = Counter(
    "drift_events_processed",
    "Total number of processed events",
)

stream_status = Gauge(
    "drift_stream_status",
    "Stream status: -1=insufficient data, 0=ok, 1=warning, 2=critical",
)

event_time_lag_seconds = Gauge(
    "drift_event_time_lag_seconds",
    "Difference in seconds between processing time and event time",
)

window_time_span_seconds = Gauge(
    "drift_window_time_span_seconds",
    "Time span in seconds covered by the current event window",
)

max_event_gap_seconds = Gauge(
    "drift_max_event_gap_seconds",
    "Maximum time gap in seconds between adjacent events in the current window",
)

invalid_event_time_rate = Gauge(
    "drift_invalid_event_time_rate",
    "Fraction of events in the current window with an invalid event time",
)

late_events = Counter(
    "drift_late_events",
    "Total number of events received later than the expected event-time boundary",
)

out_of_order_events = Counter(
    "drift_out_of_order_events",
    "Total number of events received out of event-time order",
)


# Метрики отдельных признаков
feature_status = Gauge(
    "drift_feature_status",
    "Feature drift status: -1=insufficient data, 0=ok, 1=warning, 2=critical",
    ["feature", "type"],
)

feature_psi = Gauge(
    "drift_feature_psi",
    "Population Stability Index for a feature",
    ["feature", "type"],
)

feature_missing_rate = Gauge(
    "drift_feature_missing_rate",
    "Missing value rate for a feature",
    ["feature", "type"],
)

feature_mean_zscore = Gauge(
    "drift_feature_mean_zscore",
    "Mean z-score for a numeric feature",
    ["feature", "type"],
)

feature_unseen_category_rate = Gauge(
    "drift_feature_unseen_category_rate",
    "Rate of categories absent from the reference data",
    ["feature", "type"],
)

feature_cardinality_ratio = Gauge(
    "drift_feature_cardinality_ratio",
    "Current-to-reference cardinality ratio",
    ["feature", "type"],
)

feature_cramer_v_score = Gauge(
    "drift_feature_cramer_v_score",
    "Cramer's V drift score for a categorical feature",
    ["feature", "type"],
)

feature_active_alerts = Gauge(
    "drift_feature_active_alerts",
    "Number of active alerts for a feature",
    ["feature", "type"],
)

feature_alert_info = Gauge(
    "drift_feature_alert_info",
    "Active alert for a feature: 1=active",
    ["feature", "type", "alert"],
)

metric_threshold = Gauge(
    "drift_metric_threshold",
    "Configured warning and critical thresholds for drift metrics",
    ["metric", "level"],
)


def set_critical_scenario() -> None:
    """Populate the exporter with deterministic test values."""

    overall_status.set(2)
    active_alerts.set(15)

    numeric_features = {
        "age": {
            "status": 2,
            "active_alerts": 2,
            "alerts": [
                "psi_critical",
                "mean_zscore_critical",
            ],
            "psi": 0.31,
            "missing_rate": 0.01,
            "mean_zscore": 3.4,
        },
        "income": {
            "status": 0,
            "active_alerts": 0,
            "alerts": [],
            "psi": 0.06,
            "missing_rate": 0.03,
            "mean_zscore": 0.8,
        },
        "tenure": {
            "status": 1,
            "active_alerts": 1,
            "alerts": [
                "psi_warning",
            ],
            "psi": 0.14,
            "missing_rate": 0.01,
            "mean_zscore": 1.7,
        },
        "balance": {
            "status": 2,
            "active_alerts": 2,
            "alerts": [
                "psi_critical",
                "mean_zscore_warning",
            ],
            "psi": 0.28,
            "missing_rate": 0.08,
            "mean_zscore": 2.2,
        },
        "transactions": {
            "status": 0,
            "active_alerts": 0,
            "alerts": [],
            "psi": 0.04,
            "missing_rate": 0.005,
            "mean_zscore": 0.4,
        },
        "credit_score": {
            "status": 1,
            "active_alerts": 2,
            "alerts": [
                "psi_warning",
                "mean_zscore_warning",
            ],
            "psi": 0.18,
            "missing_rate": 0.02,
            "mean_zscore": 2.4,
        },
    }
    categorical_features = {
        "country": {
            "status": 2,
            "active_alerts": 2,
            "alerts": [
                "psi_warning",
                "unseen_category_rate_critical",
            ],
            "psi": 0.22,
            "missing_rate": 0.02,
            "unseen_category_rate": 0.12,
            "cardinality_ratio": 1.5,
            "cramer_v_score": 0.32,
        },
        "device_type": {
            "status": 0,
            "active_alerts": 0,
            "alerts": [],
            "psi": 0.03,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.01,
            "cardinality_ratio": 1.0,
            "cramer_v_score": 0.05,
        },
        "channel": {
            "status": 1,
            "active_alerts": 2,
            "alerts": [
                "psi_warning",
                "unseen_category_rate_warning",
            ],
            "psi": 0.13,
            "missing_rate": 0.02,
            "unseen_category_rate": 0.07,
            "cardinality_ratio": 1.2,
            "cramer_v_score": 0.15,
        },
        "region": {
            "status": 2,
            "active_alerts": 2,
            "alerts": [
                "psi_critical",
                "unseen_category_rate_critical",
            ],
            "psi": 0.27,
            "missing_rate": 0.03,
            "unseen_category_rate": 0.14,
            "cardinality_ratio": 1.6,
            "cramer_v_score": 0.29,
        },
        "product": {
            "status": 0,
            "active_alerts": 0,
            "alerts": [],
            "psi": 0.05,
            "missing_rate": 0.00,
            "unseen_category_rate": 0.02,
            "cardinality_ratio": 1.05,
            "cramer_v_score": 0.04,
        },
        "customer_segment": {
            "status": 1,
            "active_alerts": 2,
            "alerts": [
                "psi_warning",
                "unseen_category_rate_warning",
            ],
            "psi": 0.16,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.06,
            "cardinality_ratio": 1.1,
            "cramer_v_score": 0.13,
        },
    }

    for feature, values in numeric_features.items():
        labels = {"feature": feature, "type": "numeric"}

        feature_status.labels(**labels).set(values["status"])
        feature_active_alerts.labels(**labels).set(
            values["active_alerts"]
        )
        feature_psi.labels(**labels).set(values["psi"])
        feature_missing_rate.labels(**labels).set(
            values["missing_rate"]
        )
        feature_mean_zscore.labels(**labels).set(
            values["mean_zscore"]
        )

        for alert in values["alerts"]:
            feature_alert_info.labels(
                feature=feature,
                type="numeric",
                alert=alert,
            ).set(1)

    for feature, values in categorical_features.items():
        labels = {"feature": feature, "type": "categorical"}

        feature_status.labels(**labels).set(values["status"])
        feature_active_alerts.labels(**labels).set(
            values["active_alerts"]
        )
        feature_psi.labels(**labels).set(values["psi"])
        feature_missing_rate.labels(**labels).set(
            values["missing_rate"]
        )
        feature_unseen_category_rate.labels(**labels).set(
            values["unseen_category_rate"]
        )
        feature_cardinality_ratio.labels(**labels).set(
            values["cardinality_ratio"]
        )
        feature_cramer_v_score.labels(**labels).set(
            values["cramer_v_score"]
        )
        for alert in values["alerts"]:
            feature_alert_info.labels(
                feature=feature,
                type="categorical",
                alert=alert,
            ).set(1)

    metric_threshold.labels(metric="psi", level="warning").set(0.1)
    metric_threshold.labels(metric="psi", level="critical").set(0.25)

    metric_threshold.labels(metric="mean_zscore", level="warning").set(2.0)
    metric_threshold.labels(metric="mean_zscore", level="critical").set(3.0)

    metric_threshold.labels(
        metric="unseen_category_rate",
        level="warning",
    ).set(0.05)

    metric_threshold.labels(
        metric="unseen_category_rate",
        level="critical",
    ).set(0.1)


FEATURE_LABELS = (
    *((name, "numeric") for name in (
        "age",
        "income",
        "tenure",
        "balance",
        "transactions",
        "credit_score",
    )),
    *((name, "categorical") for name in (
        "country",
        "device_type",
        "channel",
        "region",
        "product",
        "customer_segment",
    )),
)


def set_insufficient_data() -> None:
    """Expose the warm-up state until the first window is analyzed."""

    stream_status.set(-1)
    overall_status.set(-1)
    active_alerts.set(0)
    feature_alert_info.clear()

    for feature, feature_type in FEATURE_LABELS:
        labels = {"feature": feature, "type": feature_type}
        feature_status.labels(**labels).set(-1)
        feature_active_alerts.labels(**labels).set(0)


def current_stream_state(seconds_since_full: float) -> int:
    """Return the current state in a roughly 2.25-minute repeating cycle."""

    position = seconds_since_full % STATE_CYCLE_SECONDS
    if position < NORMAL_DURATION_SECONDS:
        return 0
    if position < NORMAL_DURATION_SECONDS + WARNING_DURATION_SECONDS:
        return 1
    return 2


def update_stream_health(
    status: int,
    tick: int,
    rng: random.Random,
) -> None:
    """Update technical stream metrics for one generated event."""

    stream_status.set(status)

    if status <= 0:
        event_time_lag_seconds.set(rng.uniform(0.0, 3.0))
        max_event_gap_seconds.set(rng.uniform(0.5, 2.0))
        invalid_event_time_rate.set(0)
        return

    if status == 1:
        event_time_lag_seconds.set(rng.uniform(5.0, 12.0))
        max_event_gap_seconds.set(rng.uniform(2.0, 6.0))
        invalid_event_time_rate.set(0.01 if tick % 20 < 4 else 0)
        if tick % 10 == 0:
            late_events.inc()
        if tick % 30 == 0:
            out_of_order_events.inc()
        return

    event_time_lag_seconds.set(rng.uniform(20.0, 45.0))
    max_event_gap_seconds.set(rng.uniform(8.0, 20.0))
    invalid_event_time_rate.set(rng.uniform(0.03, 0.08))
    if tick % 4 == 0:
        late_events.inc()
    if tick % 8 == 0:
        out_of_order_events.inc()


def simulate_event_stream() -> None:
    """Generate events and update stream metrics in a background thread."""

    rng = random.Random()
    current_window_size = 0
    tick = 0
    first_analysis_at: float | None = None
    last_analysis_at: float | None = None

    window_time_span_seconds.set(0)
    event_time_lag_seconds.set(0)
    max_event_gap_seconds.set(0)
    invalid_event_time_rate.set(0)
    set_insufficient_data()

    while True:
        tick += 1
        events_processed.inc()
        current_window_size += 1
        now = time.monotonic()
        window_time_span_seconds.set(
            (current_window_size - 1) * EVENT_INTERVAL_SECONDS
        )

        if last_analysis_at is not None:
            last_analysis_age_seconds.set(now - last_analysis_at)

        if current_window_size == WINDOW_SIZE:
            set_critical_scenario()
            analysis_runs.inc()
            last_analysis_at = now
            last_analysis_age_seconds.set(0)
            if first_analysis_at is None:
                first_analysis_at = now
            current_window_size = 0

        if first_analysis_at is None:
            update_stream_health(-1, tick, rng)
        else:
            status = current_stream_state(now - first_analysis_at)
            update_stream_health(status, tick, rng)

        time.sleep(EVENT_INTERVAL_SECONDS)


if __name__ == "__main__":
    set_critical_scenario()
    start_http_server(8000)

    stream_thread = threading.Thread(
        target=simulate_event_stream,
        name="mock-event-stream",
        daemon=True,
    )
    stream_thread.start()

    print("Mock exporter is running on port 8000")

    while True:
        time.sleep(60)
