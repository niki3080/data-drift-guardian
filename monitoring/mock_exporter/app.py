import random
import threading
import time
from datetime import datetime, timezone
from typing import Any

from prometheus_client import Counter, Gauge, start_http_server


EVENT_INTERVAL_SECONDS = 0.01
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

report_timestamp_seconds = Gauge(
    "drift_report_timestamp_seconds",
    "Unix timestamp carried by the latest drift report",
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


# Универсальные метрики нового drift_report
metric_value = Gauge(
    "drift_metric_value",
    "Calculated drift metric value",
    ["feature", "type", "metric"],
)

metric_status = Gauge(
    "drift_status",
    "Metric status: -1=insufficient data, 0=ok, 1=warning, 2=critical",
    ["feature", "type", "metric"],
)

feature_status = Gauge(
    "drift_status_feature",
    "Feature drift status: -1=insufficient data, 0=ok, 1=warning, 2=critical",
    ["feature", "type"],
)

metric_threshold = Gauge(
    "drift_threshold",
    "Configured warning and critical thresholds for drift metrics",
    ["metric", "level"],
)


SEVERITY_TO_CODE = {
    "insufficient_data": -1,
    "ok": 0,
    "warning": 1,
    "critical": 2,
}


def export_to_prometheus(report: dict[str, Any]) -> None:
    """Преобразует один drift_report в набор Prometheus-метрик."""

    timestamp = datetime.fromisoformat(
        report["timestamp"].replace("Z", "+00:00")
    )
    report_timestamp_seconds.set(timestamp.timestamp())
    window_size.set(report["window_size"])
    overall_status.set(SEVERITY_TO_CODE[report["overall_status"]])
    active_alerts.set(report["active_alerts"])

    metric_value.clear()
    metric_status.clear()
    feature_status.clear()
    metric_threshold.clear()

    for metric_name, levels in report["thresholds"].items():
        for level in ("warning", "critical"):
            metric_threshold.labels(
                metric=metric_name,
                level=level,
            ).set(levels[level])

    for feature_name, feature_data in report["features"].items():
        _export_feature(feature_name, feature_data)

    prediction = report.get("prediction")
    if prediction:
        _export_feature("prediction", prediction)


def _export_feature(
    feature_name: str,
    feature_data: dict[str, Any],
) -> None:
    feature_type = feature_data["type"]
    feature_labels = {"feature": feature_name, "type": feature_type}
    feature_status.labels(**feature_labels).set(
        SEVERITY_TO_CODE[feature_data["status"]]
    )

    for metric_name, result in feature_data["metrics"].items():
        labels = {**feature_labels, "metric": metric_name}
        metric_value.labels(**labels).set(result["value"])
        metric_status.labels(**labels).set(
            SEVERITY_TO_CODE[result["status"]]
        )


def set_critical_scenario() -> None:
    """Заполняет exporter небольшим согласованным drift-report."""
    thresholds = {
        "psi": {"warning": 0.1, "critical": 0.25},
        "missing_rate": {"warning": 0.05, "critical": 0.1},
        "unseen_category_rate": {"warning": 0.05, "critical": 0.1},
        "cardinality_ratio": {"warning": 1.3, "critical": 2.0},
        "js_divergence": {"warning": 0.1, "critical": 0.25},
        "wasserstein_distance": {"warning": 0.1, "critical": 0.25},
        "chi2": {"warning": 3.84, "critical": 6.63},
        "cramer_v": {"warning": 0.1, "critical": 0.25},
        "category_churn": {"warning": 0.1, "critical": 0.25},
        "kstest": {"warning": 0.1, "critical": 0.2},
        "prediction_psi": {"warning": 0.1, "critical": 0.25},
    }

    def feature_report(
        feature_type: str,
        values: dict[str, float],
    ) -> dict[str, Any]:
        results: dict[str, dict[str, Any]] = {}
        for metric_name, value in values.items():
            warning = thresholds[metric_name]["warning"]
            critical = thresholds[metric_name]["critical"]
            if value >= critical:
                status = "critical"
            elif value >= warning:
                status = "warning"
            else:
                status = "ok"

            results[metric_name] = {
                "value": value,
                "status": status,
            }

        severity = max(
            SEVERITY_TO_CODE[result["status"]]
            for result in results.values()
        )
        code_to_status = {
            code: status for status, code in SEVERITY_TO_CODE.items()
        }
        return {
            "type": feature_type,
            "status": code_to_status[severity],
            "metrics": results,
        }

    numeric_defaults = {
        "psi": 0.04,
        "missing_rate": 0.01,
        "js_divergence": 0.03,
        "wasserstein_distance": 0.04,
        "kstest": 0.05,
    }
    categorical_defaults = {
        "psi": 0.04,
        "missing_rate": 0.01,
        "unseen_category_rate": 0.01,
        "cardinality_ratio": 1.05,
        "js_divergence": 0.03,
        "chi2": 1.2,
        "cramer_v": 0.04,
        "category_churn": 0.03,
    }
    numeric_features = {
        # три алерта: critical, warning, warning
        "age": {"psi": 0.31, "missing_rate": 0.06, "wasserstein_distance": 0.15},
        # два warning-алерта
        "income": {"psi": 0.15, "kstest": 0.15},
        # один warning-алерт и один critical-алерт
        "tenure": {"js_divergence": 0.15, "wasserstein_distance": 0.30},
        # один critical-алерт
        "balance": {"missing_rate": 0.12},
        # один warning-алерт
        "transactions": {"missing_rate": 0.06},
        "credit_score": {"psi": 0.05, "missing_rate": 0.01, "wasserstein_distance": 0.05},
        "account_age_days": {"psi": 0.02, "missing_rate": 0.00, "wasserstein_distance": 0.03},
        "monthly_spend": {"psi": 0.08, "missing_rate": 0.02, "wasserstein_distance": 0.07},
        "login_count": {"psi": 0.04, "missing_rate": 0.01, "wasserstein_distance": 0.02},
        "support_tickets": {"psi": 0.06, "missing_rate": 0.03, "wasserstein_distance": 0.05},
    }
    categorical_features = {
        # один warning-алерт
        "country": {"unseen_category_rate": 0.06},
        "device_type": {"psi": 0.03, "missing_rate": 0.01, "unseen_category_rate": 0.01, "cramer_v": 0.04},
        "channel": {"psi": 0.04, "missing_rate": 0.01, "unseen_category_rate": 0.02, "cramer_v": 0.03},
        "region": {"psi": 0.06, "missing_rate": 0.02, "unseen_category_rate": 0.01, "cramer_v": 0.05},
        "product": {"psi": 0.03, "missing_rate": 0.00, "unseen_category_rate": 0.02, "cramer_v": 0.04},
        "customer_segment": {"psi": 0.07, "missing_rate": 0.01, "unseen_category_rate": 0.03, "cramer_v": 0.06},
        "plan_type": {"psi": 0.05, "missing_rate": 0.02, "unseen_category_rate": 0.01, "cramer_v": 0.04},
        "browser": {"psi": 0.04, "missing_rate": 0.01, "unseen_category_rate": 0.02, "cramer_v": 0.03},
        "payment_method": {"psi": 0.06, "missing_rate": 0.02, "unseen_category_rate": 0.01, "cramer_v": 0.05},
        "acquisition_source": {"psi": 0.03, "missing_rate": 0.01, "unseen_category_rate": 0.02, "cramer_v": 0.04},
    }
    features = {
        **{
            name: feature_report(
                "numeric",
                {**numeric_defaults, **values},
            )
            for name, values in numeric_features.items()
        },
        **{
            name: feature_report(
                "categorical",
                {**categorical_defaults, **values},
            )
            for name, values in categorical_features.items()
        },
    }
    # включить предикт
    prediction = feature_report("numeric", {
        "prediction_psi": 0.11,
        "missing_rate": 0.01,
        "js_divergence": 0.06,
        "wasserstein_distance": 0.08,
        "kstest": 0.08,
    })

    all_features = (*features.values(), prediction)

    # отключить предикт
    # prediction = None
    # all_features = tuple(features.values())


    alert_count = sum(
        result["status"] != "ok"
        for feature in all_features
        for result in feature["metrics"].values()
    )
    overall_severity = max(
        SEVERITY_TO_CODE[feature["status"]]
        for feature in all_features
    )
    code_to_status = {
        code: status for status, code in SEVERITY_TO_CODE.items()
    }
    report = {
        "timestamp": (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "window_size": WINDOW_SIZE,
        "overall_status": code_to_status[overall_severity],
        "active_alerts": alert_count,
        "thresholds": thresholds,
        "features": features,
        "prediction": prediction,
    }
    export_to_prometheus(report)


FEATURE_LABELS = (
    *((name, "numeric") for name in (
        "age",
        "income",
        "tenure",
        "balance",
        "transactions",
        "credit_score",
        "account_age_days",
        "monthly_spend",
        "login_count",
        "support_tickets",
    )),
    *((name, "categorical") for name in (
        "country",
        "device_type",
        "channel",
        "region",
        "product",
        "customer_segment",
        "plan_type",
        "browser",
        "payment_method",
        "acquisition_source",
    )),
    ("prediction", "numeric"),
)


def set_insufficient_data() -> None:
    """Показывает состояние прогрева до анализа первого полного окна."""

    stream_status.set(-1)
    overall_status.set(-1)
    active_alerts.set(0)
    report_timestamp_seconds.set(-1)
    metric_value.clear()
    metric_status.clear()
    feature_status.clear()

    for feature, feature_type in FEATURE_LABELS:
        labels = {"feature": feature, "type": feature_type}
        feature_status.labels(**labels).set(-1)


def current_stream_state(seconds_since_full: float) -> int:
    """Возвращает состояние текущей фазы повторяющегося demo-цикла."""

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
    """Обновляет технические stream-метрики для одного demo-события."""

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
    """Генерирует события и обновляет stream-метрики в фоновом потоке."""

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
