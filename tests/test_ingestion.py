import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.consumer import wait_for_kafka_topic
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.runtime import AvThresholds, RuntimeContext
from drift_guardian.ingestion.stream_metrics import (
    StreamThresholds,
    StreamTracker,
    ThresholdPair,
    stream_thresholds_from_config,
)
from drift_guardian.ingestion.window import WindowBuffer, analyze_window


class _TopicMetadata:
    def __init__(self, error=None) -> None:
        self.error = error


class _ClusterMetadata:
    def __init__(self, topics) -> None:
        self.topics = topics


class _FakeAdminClient:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = []

    def list_topics(self, *, topic, timeout):
        self.calls.append((topic, timeout))
        return next(self.responses)


def test_wait_for_kafka_topic_retries_until_topic_exists() -> None:
    fake_admin = _FakeAdminClient(
        [
            _ClusterMetadata({}),
            _ClusterMetadata({"features-stream": _TopicMetadata()}),
        ]
    )
    configs = []
    sleeps = []
    times = iter([0.0, 0.0, 1.0])

    wait_for_kafka_topic(
        "kafka:19092",
        "features-stream",
        timeout_seconds=5.0,
        retry_interval_seconds=0.5,
        admin_client_factory=lambda config: configs.append(config) or fake_admin,
        monotonic=lambda: next(times),
        sleep=sleeps.append,
    )

    assert configs[0]["bootstrap.servers"] == "kafka:19092"
    assert configs[0]["log_level"] == 0
    assert len(fake_admin.calls) == 2
    assert sleeps == [0.5]


def test_wait_for_kafka_topic_has_bounded_timeout() -> None:
    fake_admin = _FakeAdminClient([_ClusterMetadata({})])
    times = iter([0.0, 0.0, 2.0])

    with pytest.raises(TimeoutError, match="Kafka startup timeout"):
        wait_for_kafka_topic(
            "kafka:19092",
            "features-stream",
            timeout_seconds=1.0,
            retry_interval_seconds=0.1,
            admin_client_factory=lambda _: fake_admin,
            monotonic=lambda: next(times),
            sleep=lambda _: None,
        )


def test_event_contract() -> None:
    event = KafkaEvent.from_dict(
        {
            "event_id": 1,
            "event_time": "2026-09-13T11:32:00Z",
            "age": 45,
            "prediction_score": 0.72,
        }
    )
    assert event.features["age"] == 45

    with pytest.raises(InvalidEventTime):
        KafkaEvent.from_dict(
            {
                "event_id": 2,
                "event_time": "2026-09-13T11:32:00",
                "age": 45,
            }
        )


def test_windows_are_non_overlapping() -> None:
    window = WindowBuffer(3)
    now = datetime.now(UTC)
    for event_id in range(1, 4):
        window.append(
            KafkaEvent.from_dict(
                {
                    "event_id": event_id,
                    "event_time": now.isoformat(),
                    "value": event_id,
                }
            )
        )

    assert window.is_full
    assert window.to_dataframe()["value"].tolist() == [1, 2, 3]

    with pytest.raises(RuntimeError):
        window.append(
            KafkaEvent.from_dict(
                {
                    "event_id": 4,
                    "event_time": now.isoformat(),
                    "value": 4,
                }
            )
        )

    window.clear()
    assert not window.is_full
    assert len(window) == 0


def test_analyzer_runs_only_for_full_window() -> None:
    calls = []
    window = WindowBuffer(2)
    now = datetime.now(UTC)

    def analyzer(current_df):
        calls.append(current_df.copy())
        return {"overall_status": "ok", "features": {}}

    window.append(
        KafkaEvent.from_dict(
            {"event_id": 1, "event_time": now.isoformat(), "value": 1}
        )
    )
    assert analyze_window(window, analyzer) is None

    window.append(
        KafkaEvent.from_dict(
            {"event_id": 2, "event_time": now.isoformat(), "value": 2}
        )
    )
    report = analyze_window(window, analyzer)

    assert len(calls) == 1
    assert calls[0]["value"].tolist() == [1, 2]
    assert report["overall_status"] == "ok"


def test_stream_quality_is_window_local_and_worst_status_wins() -> None:
    thresholds = StreamThresholds(
        invalid_event_time_rate=ThresholdPair(0.2, 0.5),
    )
    tracker = StreamTracker(
        late_event_threshold_seconds=1000,
        thresholds=thresholds,
    )
    window = WindowBuffer(2)
    now = datetime.now(UTC)

    tracker.record_invalid_event_time()
    for event_id in (1, 2):
        event = KafkaEvent.from_dict(
            {
                "event_id": event_id,
                "event_time": (now + timedelta(seconds=event_id)).isoformat(),
                "value": event_id,
            }
        )
        window.append(event)
        tracker.observe(event)

    warmup = tracker.snapshot(window.event_times(), ready=False)
    assert warmup.status == -1

    snapshot = tracker.snapshot(window.event_times(), ready=True)
    assert snapshot.invalid_event_time_rate == pytest.approx(1 / 3)
    assert snapshot.status == 1

    tracker.reset_window()
    window.clear()
    next_event = KafkaEvent.from_dict(
        {
            "event_id": 3,
            "event_time": (now + timedelta(seconds=3)).isoformat(),
            "value": 3,
        }
    )
    window.append(next_event)
    tracker.observe(next_event)

    snapshot = tracker.snapshot(window.event_times(), ready=True)
    assert snapshot.invalid_event_time_rate == 0.0
    assert snapshot.status == 0


def test_runtime_av_schedule_uses_core_config_as_source_of_truth() -> None:
    core = SimpleNamespace(
        config=SimpleNamespace(
            adversarial_validation=SimpleNamespace(
                enabled=True,
                interval_minutes=30,
            )
        )
    )
    runtime = RuntimeContext(
        core=core,
        dataset_name="demo",
        profile_created_at="2026-09-24T00:00:00Z",
        adversarial_thresholds=AvThresholds(0.60, 0.75),
        adversarial_top_features=10,
    )

    assert runtime.av_due(100.0) is True
    runtime._last_av_monotonic = 100.0
    assert runtime.av_due(100.0 + 30 * 60 - 1) is False
    assert runtime.av_due(100.0 + 30 * 60) is True

    core.config.adversarial_validation.enabled = False
    assert runtime.av_due(10_000.0) is False


def test_stream_thresholds_are_resolved_from_existing_config_object() -> None:
    config = SimpleNamespace(
        stream_drift=SimpleNamespace(
            drift_event_time_lag_seconds=SimpleNamespace(warning=30, critical=120),
            drift_window_time_span_seconds=None,
            drift_max_event_gap_seconds=None,
            drift_invalid_event_time_rate=None,
            drift_late_event_rate=SimpleNamespace(warning=0.01, critical=0.05),
        )
    )

    thresholds = stream_thresholds_from_config(config)

    assert thresholds.event_time_lag_seconds == ThresholdPair(30.0, 120.0)
    assert thresholds.late_event_rate == ThresholdPair(0.01, 0.05)
    assert thresholds.window_time_span_seconds is None


def test_stream_status_is_insufficient_without_configured_thresholds() -> None:
    tracker = StreamTracker(late_event_threshold_seconds=1)
    event = KafkaEvent.from_dict(
        {
            "event_id": 1,
            "event_time": datetime.now(UTC).isoformat(),
            "value": 1,
        }
    )
    tracker.observe(event)

    snapshot = tracker.snapshot([event.event_time], ready=True)

    assert snapshot.status == -1


def test_stream_status_uses_current_window_late_event_rate() -> None:
    tracker = StreamTracker(
        late_event_threshold_seconds=1,
        thresholds=StreamThresholds(
            event_time_lag_seconds=ThresholdPair(100, 200),
            late_event_rate=ThresholdPair(0.5, 0.75),
        ),
    )
    now = datetime.now(UTC)
    tracker.record_invalid_event_time()

    first = KafkaEvent.from_dict(
        {
            "event_id": 1,
            "event_time": (now - timedelta(seconds=5)).isoformat(),
            "value": 1,
        }
    )
    tracker.observe(first)
    assert tracker.snapshot([first.event_time], ready=False).status == -1
    critical = tracker.snapshot([first.event_time], ready=True)
    assert critical.late_event_rate == 1.0
    assert critical.status == 2

    on_time = KafkaEvent.from_dict(
        {
            "event_id": 2,
            "event_time": now.isoformat(),
            "value": 2,
        }
    )
    tracker.observe(on_time)
    warning = tracker.snapshot(
        [first.event_time, on_time.event_time],
        ready=True,
    )
    assert warning.late_event_rate == 0.5
    assert warning.status == 1

    tracker.reset_window()
    recovered = tracker.snapshot([], ready=True)
    assert recovered.late_event_rate == 0.0
    assert recovered.status == 0

    second_late = KafkaEvent.from_dict(
        {
            "event_id": 3,
            "event_time": (now - timedelta(seconds=4)).isoformat(),
            "value": 3,
        }
    )
    tracker.observe(second_late)
    next_window = tracker.snapshot([second_late.event_time], ready=True)
    assert next_window.late_event_rate == 1.0
    assert next_window.late_events_total == 2
    assert next_window.status == 2


def test_lifetime_counters_survive_window_reset() -> None:
    tracker = StreamTracker(late_event_threshold_seconds=1)
    now = datetime.now(UTC)
    first = KafkaEvent.from_dict(
        {
            "event_id": 1,
            "event_time": (now - timedelta(seconds=5)).isoformat(),
            "value": 1,
        }
    )
    second = KafkaEvent.from_dict(
        {
            "event_id": 2,
            "event_time": (now - timedelta(seconds=10)).isoformat(),
            "value": 2,
        }
    )
    tracker.observe(first)
    tracker.observe(second)

    before = tracker.snapshot([first.event_time, second.event_time], ready=True)
    assert before.late_events_total == 2
    assert before.out_of_order_events_total == 1

    tracker.reset_window()
    after = tracker.snapshot([], ready=True)
    assert after.late_events_total == 2
    assert after.out_of_order_events_total == 1
    assert after.invalid_event_time_rate == 0.0
    assert after.late_event_rate == 0.0


def test_prometheus_exporter_tracks_current_window_events() -> None:
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)

    initial_metrics = generate_latest(registry).decode("utf-8")
    assert "drift_current_window_events 0.0" in initial_metrics
    assert "drift_late_event_rate 0.0" in initial_metrics

    exporter.set_current_window_events(346)
    metrics = generate_latest(registry).decode("utf-8")
    assert "drift_current_window_events 346.0" in metrics

    exporter.set_current_window_events(0)
    reset_metrics = generate_latest(registry).decode("utf-8")
    assert "drift_current_window_events 0.0" in reset_metrics

    with pytest.raises(ValueError, match="non-negative"):
        exporter.set_current_window_events(-1)


def test_reference_profile_exports_only_agreed_metadata() -> None:
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)
    exporter.update_reference_profile(
        {
            "dataset_name": "demo",
            "profile_created_at": "2026-09-16T10:00:00Z",
            "sample_size": 10000,
            "profile_name": "unused",
            "profile_version": "unused",
        }
    )

    metrics = generate_latest(registry).decode()

    assert (
        'drift_reference_profile_info{dataset_name="demo",'
        'profile_created_at="2026-09-16T10:00:00Z"} 1.0'
        in metrics
    )
    assert "profile_name" not in metrics
    assert "profile_version" not in metrics
    assert "drift_reference_sample_size 10000.0" in metrics


def test_exporter_supports_generic_report_contract() -> None:
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)
    exporter.set_window_size(1000)
    exporter.update_stream_thresholds(
        StreamThresholds(
            invalid_event_time_rate=ThresholdPair(0.05, 0.1),
            max_event_gap_seconds=ThresholdPair(10.0, 30.0),
        )
    )
    exporter.update_report(
        {
            "timestamp": "2026-09-16T10:30:00Z",
            "window_size": 1000,
            "overall_status": "critical",
            "active_alerts": 1,
            "features": {
                "age": {
                    "type": "numeric",
                    "status": "critical",
                    "metrics": {
                        "psi": {
                            "value": 0.31,
                            "warning": 0.1,
                            "critical": 0.25,
                            "status": "critical",
                        },
                        "kstest": {
                            "value": 0.2,
                            "warning": 0.1,
                            "critical": 0.3,
                            "status": "warning",
                        },
                    },
                }
            },
            "prediction": {
                "type": "numeric",
                "status": "warning",
                "metrics": {
                    "psi": {
                        "value": 0.14,
                        "warning": 0.1,
                        "critical": 0.25,
                        "status": "warning",
                    }
                },
            },
        }
    )
    exporter.record_analysis_run()

    metrics = generate_latest(registry).decode()
    for sample in (
        'drift_status_feature{feature="age",type="numeric"} 2.0',
        'drift_metric_value{feature="age",metric="psi",type="numeric"} 0.31',
        'drift_status{feature="age",metric="psi",type="numeric"} 2.0',
        'drift_resolved_threshold{feature="age",level="warning",metric="psi",type="numeric"} 0.1',
        'drift_metric_value{feature="prediction",'
        'metric="prediction_psi",type="numeric"} 0.14',
        'drift_resolved_threshold{feature="prediction",level="warning",metric="prediction_psi",type="numeric"} 0.1',
        "drift_analysis_runs_total 1.0",
        'drift_stream_threshold{level="warning",'
        'metric="drift_invalid_event_time_rate"} 0.05',
        'drift_stream_threshold{level="critical",'
        'metric="drift_max_event_gap_seconds"} 30.0',
    ):
        assert sample in metrics

    assert re.search(
        r"drift_report_timestamp_seconds [1-9][0-9]*(?:\.[0-9]+)?(?:e\+[0-9]+)?",
        metrics,
    )


def test_metric_status_falls_back_to_feature_status_and_prediction_resets() -> None:
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)

    exporter.update_report(
        {
            "overall_status": "warning",
            "features": {
                "age": {
                    "type": "numeric",
                    "metrics": {
                        "psi": {"value": 0.15, "status": "warning"}
                    },
                }
            },
            "prediction": {
                "type": "numeric",
                "status": "warning",
                "metrics": {"psi": {"value": 0.15, "status": "warning"}},
            },
            "thresholds": {"psi": {"warning": 0.1, "critical": 0.25}},
        }
    )
    metrics = generate_latest(registry).decode()
    assert 'drift_status_feature{feature="age",type="numeric"} 1.0' in metrics
    assert (
        'drift_status_feature{feature="prediction",type="numeric"} 1.0'
        in metrics
    )

    exporter.update_report(
        {
            "overall_status": "ok",
            "features": {},
        }
    )
    metrics = generate_latest(registry).decode()
    assert 'feature="prediction"' not in metrics


def test_exporter_accepts_flat_core_report() -> None:
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)
    exporter.update_report(
        {
            "overall_status": "critical",
            "active_alerts": 2,
            "window_size": 1000,
            "thresholds": {"psi": {"warning": 0.1, "critical": 0.25}},
            "features": {
                "age": {
                    "type": "numeric",
                    "status": "critical",
                    "metrics": {"psi": 0.31, "missing_rate": 0.01},
                    "alerts": ["psi_critical"],
                }
            },
        }
    )

    metrics = generate_latest(registry).decode()
    assert (
        'drift_metric_value{feature="age",metric="psi",type="numeric"} 0.31'
        in metrics
    )
    assert (
        'drift_status{feature="age",metric="psi",type="numeric"} 2.0'
        in metrics
    )
    assert 'drift_threshold{level="critical",metric="psi"} 0.25' in metrics
