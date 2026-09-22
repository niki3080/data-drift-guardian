import random
import threading
import time
from datetime import datetime, timezone
from typing import Any

from prometheus_client import Counter, Gauge, Info, start_http_server

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
ANALYSIS_SEQUENCE = 0
PREVIOUS_AV_IMPORTANCE: dict[str, float] = {}
AV_WARNING_AUC = 0.60
AV_CRITICAL_AUC = 0.75
STREAM_LAG_WARNING_SECONDS = 30
STREAM_LAG_CRITICAL_SECONDS = 120
STREAM_LATE_RATE_WARNING = 0.01
STREAM_LATE_RATE_CRITICAL = 0.05
WARNING_LATE_EVENT_INTERVAL = 40
CRITICAL_LATE_EVENT_INTERVAL = 12
WARNING_OUT_OF_ORDER_EVENT_INTERVAL = 100
CRITICAL_OUT_OF_ORDER_EVENT_INTERVAL = 25


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

current_window_events = Gauge(
    "drift_current_window_events",
    "Number of events currently collected in the analysis window",
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

reference_profile_info = Info(
    "drift_reference_profile",
    "Metadata of the active reference sample",
)
reference_profile_info.info(
    {
        "dataset_name": "demo_reference",
        "profile_created_at": "2026-09-20T00:00:00Z",
    }
)

window_size.set(WINDOW_SIZE)
current_window_events.set(0)
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

late_event_rate = Gauge(
    "drift_late_event_rate",
    "Fraction of late events in the current analysis window",
)

stream_threshold = Gauge(
    "drift_stream_threshold",
    "Configured warning and critical stream-quality thresholds",
    ["metric", "level"],
)
stream_threshold.labels(
    metric="drift_event_time_lag_seconds",
    level="warning",
).set(STREAM_LAG_WARNING_SECONDS)
stream_threshold.labels(
    metric="drift_event_time_lag_seconds",
    level="critical",
).set(STREAM_LAG_CRITICAL_SECONDS)
stream_threshold.labels(
    metric="drift_late_event_rate",
    level="warning",
).set(STREAM_LATE_RATE_WARNING)
stream_threshold.labels(
    metric="drift_late_event_rate",
    level="critical",
).set(STREAM_LATE_RATE_CRITICAL)

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

av_status = Gauge(
    "drift_av_status",
    "Adversarial validation status: -1=not configured, 0=ok, 1=warning, 2=critical",
)
av_available = Gauge(
    "drift_av_available",
    "Whether an adversarial validation result is available: 0=no, 1=yes",
)
av_roc_auc = Gauge(
    "drift_av_roc_auc",
    "Adversarial validation ROC AUC",
)
av_roc_auc_cv_std = Gauge(
    "drift_av_roc_auc_cv_std",
    "Standard deviation of fold ROC AUC values in adversarial validation",
)
av_roc_auc_cv_mean = Gauge(
    "drift_av_roc_auc_cv_mean",
    "Mean ROC AUC across adversarial-validation CV folds",
)
av_roc_auc_cv_min = Gauge(
    "drift_av_roc_auc_cv_min",
    "Worst ROC AUC across adversarial-validation CV folds",
)
av_roc_auc_cv_max = Gauge(
    "drift_av_roc_auc_cv_max",
    "Best ROC AUC across adversarial-validation CV folds",
)
av_driver_consistency = Gauge(
    "drift_av_driver_consistency",
    "Feature-importance consistency across adversarial-validation CV folds",
)
av_driver_similarity_previous = Gauge(
    "drift_av_driver_similarity_previous",
    "Cosine similarity of AV feature importance to the previous completed AV run",
)
av_top1_importance_share = Gauge(
    "drift_av_top1_importance_share",
    "Share of total AV feature importance explained by the strongest driver",
)
av_top3_importance_share = Gauge(
    "drift_av_top3_importance_share",
    "Share of total AV feature importance explained by the top three drivers",
)
av_timestamp_seconds = Gauge(
    "drift_av_timestamp_seconds",
    "Timestamp of the latest adversarial validation run",
)
av_last_run_timestamp_seconds = Gauge(
    "drift_av_last_run_timestamp_seconds",
    "Unix timestamp of the latest adversarial validation run",
)
av_dataset_size = Gauge(
    "drift_av_dataset_size",
    "Balanced rows per dataset used by adversarial validation",
)
av_reference_rows = Gauge(
    "drift_av_reference_rows",
    "Reference rows available to adversarial validation",
)
av_current_rows = Gauge(
    "drift_av_current_rows",
    "Current rows available to adversarial validation",
)
av_features_evaluated = Gauge(
    "drift_av_features_evaluated",
    "Number of features evaluated by adversarial validation",
)
av_sample_fraction = Gauge(
    "drift_av_sample_fraction",
    "Fraction of each source dataset used in the balanced AV sample",
    ["dataset"],
)
av_feature_importance = Gauge(
    "drift_av_feature_importance",
    "Top adversarial-validation feature importance",
    ["feature", "rank"],
)
av_threshold = Gauge(
    "drift_av_threshold",
    "Configured adversarial-validation ROC-AUC thresholds",
    ["level"],
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


def export_adversarial_validation(timestamp: datetime) -> None:
    """Публикует demo-результат AV для разработки Grafana-панели."""
    auc_values = (0.54, 0.66, 0.80)
    roc_auc = auc_values[(ANALYSIS_SEQUENCE - 1) % len(auc_values)]
    if roc_auc >= AV_CRITICAL_AUC:
        status = 2
    elif roc_auc >= AV_WARNING_AUC:
        status = 1
    else:
        status = 0

    av_available.set(1)
    av_status.set(status)
    av_roc_auc.set(roc_auc)
    scenario_index = (ANALYSIS_SEQUENCE - 1) % len(auc_values)
    auc_std_values = (0.008, 0.018, 0.032)
    auc_min_values = (0.529, 0.638, 0.761)
    auc_max_values = (0.551, 0.682, 0.839)
    driver_consistency_values = (0.94, 0.86, 0.72)
    av_roc_auc_cv_std.set(auc_std_values[scenario_index])
    av_roc_auc_cv_mean.set(roc_auc)
    av_roc_auc_cv_min.set(auc_min_values[scenario_index])
    av_roc_auc_cv_max.set(auc_max_values[scenario_index])
    av_driver_consistency.set(driver_consistency_values[scenario_index])
    av_timestamp_seconds.set(timestamp.timestamp())
    av_last_run_timestamp_seconds.set(timestamp.timestamp())
    av_reference_rows.set(10_000)
    av_current_rows.set(WINDOW_SIZE)
    av_dataset_size.set(WINDOW_SIZE)
    av_features_evaluated.set(len(FEATURE_LABELS) - 1)
    av_sample_fraction.clear()
    av_sample_fraction.labels(dataset="reference").set(WINDOW_SIZE / 10_000)
    av_sample_fraction.labels(dataset="current").set(1.0)
    av_threshold.clear()
    av_threshold.labels(level="warning").set(AV_WARNING_AUC)
    av_threshold.labels(level="critical").set(AV_CRITICAL_AUC)
    av_feature_importance.clear()

    top_feature_scenarios = (
        (
            ("age", 0.29),
            ("income", 0.21),
            ("country", 0.14),
            ("balance", 0.10),
            ("monthly_spend", 0.08),
            ("device_type", 0.06),
            ("transactions", 0.04),
            ("region", 0.03),
            ("credit_score", 0.03),
            ("channel", 0.02),
        ),
        (
            ("income", 0.25),
            ("age", 0.19),
            ("monthly_spend", 0.15),
            ("country", 0.10),
            ("balance", 0.08),
            ("transactions", 0.07),
            ("device_type", 0.05),
            ("credit_score", 0.04),
            ("region", 0.04),
            ("channel", 0.03),
        ),
        (
            ("country", 0.22),
            ("income", 0.20),
            ("age", 0.16),
            ("device_type", 0.12),
            ("monthly_spend", 0.09),
            ("balance", 0.07),
            ("region", 0.05),
            ("transactions", 0.04),
            ("credit_score", 0.03),
            ("channel", 0.02),
        ),
    )
    top_features = top_feature_scenarios[scenario_index]
    current_importance = dict(top_features)
    global PREVIOUS_AV_IMPORTANCE
    if PREVIOUS_AV_IMPORTANCE:
        features = set(current_importance) | set(PREVIOUS_AV_IMPORTANCE)
        dot = sum(
            current_importance.get(name, 0.0) * PREVIOUS_AV_IMPORTANCE.get(name, 0.0)
            for name in features
        )
        current_norm = sum(current_importance.get(name, 0.0) ** 2 for name in features) ** 0.5
        previous_norm = sum(PREVIOUS_AV_IMPORTANCE.get(name, 0.0) ** 2 for name in features) ** 0.5
        similarity = dot / (current_norm * previous_norm) if current_norm and previous_norm else 0.0
        av_driver_similarity_previous.set(similarity)
    else:
        av_driver_similarity_previous.set(-1)
    PREVIOUS_AV_IMPORTANCE = current_importance
    av_top1_importance_share.set(top_features[0][1])
    av_top3_importance_share.set(sum(value for _, value in top_features[:3]))
    for rank, (feature, importance) in enumerate(top_features, start=1):
        av_feature_importance.labels(
            feature=feature,
            rank=str(rank),
        ).set(importance)


def set_critical_scenario() -> None:
    """Заполняет exporter согласованным demo drift-report."""
    global ANALYSIS_SEQUENCE
    ANALYSIS_SEQUENCE += 1
    thresholds = {
        "psi": {"warning": 0.1, "critical": 0.25},
        "missing_rate": {"warning": 0.02, "critical": 0.05},
        "unseen_category_rate": {"warning": 0.05, "critical": 0.1},
        "cardinality_ratio": {"warning": 0.3, "critical": 0.6},
        "js_divergence": {"warning": 0.1, "critical": 0.25},
        "wasserstein_distance": {"warning": 0.1, "critical": 0.25},
        "chi2": {"warning": 0.05, "critical": 0.01},
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
            phase = ((ANALYSIS_SEQUENCE + len(metric_name)) % 5) - 2
            value = max(0.0, value * (1.0 + phase * 0.03))
            warning = thresholds[metric_name]["warning"]
            critical = thresholds[metric_name]["critical"]
            if metric_name == "chi2":
                if value <= critical:
                    status = "critical"
                elif value <= warning:
                    status = "warning"
                else:
                    status = "ok"
            elif value >= critical:
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
        "cardinality_ratio": 0.05,
        "js_divergence": 0.03,
        "chi2": 0.8,
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
        "country": {
            "unseen_category_rate": 0.06,
            "cardinality_ratio": 0.35,
            "chi2": 0.03,
        },
        "device_type": {
            "psi": 0.03,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.01,
            "chi2": 0.005,
            "cramer_v": 0.04,
        },
        "channel": {
            "psi": 0.04,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.02,
            "cramer_v": 0.03,
        },
        "region": {
            "psi": 0.06,
            "missing_rate": 0.02,
            "unseen_category_rate": 0.01,
            "cramer_v": 0.05,
        },
        "product": {
            "psi": 0.03,
            "missing_rate": 0.00,
            "unseen_category_rate": 0.02,
            "cardinality_ratio": 0.65,
            "cramer_v": 0.04,
        },
        "customer_segment": {
            "psi": 0.07,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.03,
            "cramer_v": 0.06,
        },
        "plan_type": {
            "psi": 0.05,
            "missing_rate": 0.02,
            "unseen_category_rate": 0.01,
            "cramer_v": 0.04,
        },
        "browser": {
            "psi": 0.04,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.02,
            "cramer_v": 0.03,
        },
        "payment_method": {
            "psi": 0.06,
            "missing_rate": 0.02,
            "unseen_category_rate": 0.01,
            "cramer_v": 0.05,
        },
        "acquisition_source": {
            "psi": 0.03,
            "missing_rate": 0.01,
            "unseen_category_rate": 0.02,
            "cramer_v": 0.04,
        },
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

    # Core считает overall status и active alerts по feature-блоку.
    # Prediction экспортируется отдельно и не повышает общий status.
    alert_count = sum(
        feature["status"] == "critical"
        for feature in features.values()
    )
    overall_severity = max(
        SEVERITY_TO_CODE[feature["status"]]
        for feature in features.values()
    )
    code_to_status = {
        code: status for status, code in SEVERITY_TO_CODE.items()
    }
    report_time = datetime.now(timezone.utc)
    report = {
        "timestamp": report_time.isoformat().replace("+00:00", "Z"),
        "window_size": WINDOW_SIZE,
        "overall_status": code_to_status[overall_severity],
        "active_alerts": alert_count,
        "thresholds": thresholds,
        "features": features,
        "prediction": prediction,
    }
    export_to_prometheus(report)
    export_adversarial_validation(report_time)


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
    av_status.set(-1)
    av_available.set(0)
    av_roc_auc.set(-1)
    av_timestamp_seconds.set(-1)
    av_last_run_timestamp_seconds.set(-1)
    av_dataset_size.set(0)
    av_reference_rows.set(0)
    av_current_rows.set(0)
    av_features_evaluated.set(0)
    av_sample_fraction.clear()
    av_threshold.clear()
    av_feature_importance.clear()

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


def update_window_timing(
    status: int,
    rng: random.Random,
    *,
    window_events_count: int,
    window_max_event_gap: float,
) -> tuple[float, float]:
    """Обновляет согласованные span и max gap текущего demo-окна."""

    candidate_gap = 0.0
    if window_events_count > 1:
        if status <= 0:
            candidate_gap = rng.uniform(0.5, 2.0)
        elif status == 1:
            candidate_gap = rng.uniform(2.0, 6.0)
        else:
            candidate_gap = rng.uniform(8.0, 20.0)

    current_max_gap = max(window_max_event_gap, candidate_gap)
    nominal_span = (window_events_count - 1) * EVENT_INTERVAL_SECONDS
    current_span = nominal_span + max(
        current_max_gap - EVENT_INTERVAL_SECONDS,
        0.0,
    )

    window_time_span_seconds.set(current_span)
    max_event_gap_seconds.set(current_max_gap)
    return current_span, current_max_gap


def update_stream_health(
    status: int,
    tick: int,
    rng: random.Random,
    *,
    window_events_count: int,
    window_late_events: int,
) -> int:
    """Обновляет stream-метрики и возвращает число late-событий окна."""

    if status <= 0:
        event_time_lag_seconds.set(rng.uniform(0.0, 3.0))
        invalid_event_time_rate.set(0)
        is_late = False
        is_out_of_order = False
    elif status == 1:
        event_time_lag_seconds.set(rng.uniform(35.0, 90.0))
        invalid_event_time_rate.set(0.01 if tick % 20 < 4 else 0)
        is_late = tick % WARNING_LATE_EVENT_INTERVAL == 0
        is_out_of_order = tick % WARNING_OUT_OF_ORDER_EVENT_INTERVAL == 0
    else:
        event_time_lag_seconds.set(rng.uniform(130.0, 240.0))
        invalid_event_time_rate.set(rng.uniform(0.03, 0.08))
        is_late = tick % CRITICAL_LATE_EVENT_INTERVAL == 0
        is_out_of_order = (
            tick % CRITICAL_OUT_OF_ORDER_EVENT_INTERVAL == 0
        )

    if is_late:
        late_events.inc()
        window_late_events += 1
    if is_out_of_order:
        out_of_order_events.inc()

    current_late_rate = window_late_events / window_events_count
    late_event_rate.set(current_late_rate)

    calculated_status = status
    if status >= 0:
        if current_late_rate >= STREAM_LATE_RATE_CRITICAL:
            calculated_status = max(calculated_status, 2)
        elif current_late_rate >= STREAM_LATE_RATE_WARNING:
            calculated_status = max(calculated_status, 1)
    stream_status.set(calculated_status)

    return window_late_events


def simulate_event_stream() -> None:
    """Генерирует события и обновляет stream-метрики в фоновом потоке."""

    rng = random.Random()
    current_window_size = 0
    current_window_late_events = 0
    current_window_max_event_gap = 0.0
    tick = 0
    first_analysis_at: float | None = None
    last_analysis_at: float | None = None

    window_time_span_seconds.set(0)
    event_time_lag_seconds.set(0)
    max_event_gap_seconds.set(0)
    invalid_event_time_rate.set(0)
    late_event_rate.set(0)
    current_window_events.set(0)
    set_insufficient_data()

    while True:
        tick += 1
        events_processed.inc()
        current_window_size += 1
        current_window_events.set(current_window_size)
        now = time.monotonic()

        if last_analysis_at is not None:
            last_analysis_age_seconds.set(now - last_analysis_at)

        if first_analysis_at is None:
            status = -1
        else:
            status = current_stream_state(now - first_analysis_at)

        _, current_window_max_event_gap = update_window_timing(
            status,
            rng,
            window_events_count=current_window_size,
            window_max_event_gap=current_window_max_event_gap,
        )
        current_window_late_events = update_stream_health(
            status,
            tick,
            rng,
            window_events_count=current_window_size,
            window_late_events=current_window_late_events,
        )

        if current_window_size == WINDOW_SIZE:
            set_critical_scenario()
            analysis_runs.inc()
            last_analysis_at = now
            last_analysis_age_seconds.set(0)
            if first_analysis_at is None:
                first_analysis_at = now
            current_window_size = 0
            current_window_late_events = 0
            current_window_max_event_gap = 0.0
            current_window_events.set(0)
            window_time_span_seconds.set(0)
            max_event_gap_seconds.set(0)
            invalid_event_time_rate.set(0)
            late_event_rate.set(0)

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
