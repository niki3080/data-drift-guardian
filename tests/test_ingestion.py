import unittest
from datetime import UTC, datetime, timedelta

from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.consumer import wait_for_kafka_topic
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.stream_metrics import (
    StreamThresholds,
    StreamTracker,
    ThresholdPair,
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


class IngestionTest(unittest.TestCase):
    def test_wait_for_kafka_topic_retries_until_topic_exists(self) -> None:
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

        self.assertEqual(configs[0]["bootstrap.servers"], "kafka:19092")
        self.assertEqual(configs[0]["log_level"], 0)
        self.assertEqual(len(fake_admin.calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_wait_for_kafka_topic_has_bounded_timeout(self) -> None:
        fake_admin = _FakeAdminClient([_ClusterMetadata({})])
        times = iter([0.0, 0.0, 2.0])

        with self.assertRaisesRegex(TimeoutError, "Kafka startup timeout"):
            wait_for_kafka_topic(
                "kafka:19092",
                "features-stream",
                timeout_seconds=1.0,
                retry_interval_seconds=0.1,
                admin_client_factory=lambda _: fake_admin,
                monotonic=lambda: next(times),
                sleep=lambda _: None,
            )

    def test_event_contract(self) -> None:
        event = KafkaEvent.from_dict(
            {
                "event_id": 1,
                "event_time": "2026-09-13T11:32:00Z",
                "age": 45,
                "prediction_score": 0.72,
            }
        )
        self.assertEqual(event.features["age"], 45)

        with self.assertRaises(InvalidEventTime):
            KafkaEvent.from_dict(
                {
                    "event_id": 2,
                    "event_time": "2026-09-13T11:32:00",
                    "age": 45,
                }
            )

    def test_windows_are_non_overlapping(self) -> None:
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

        self.assertTrue(window.is_full)
        self.assertEqual(window.to_dataframe()["value"].tolist(), [1, 2, 3])
        with self.assertRaises(RuntimeError):
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
        self.assertFalse(window.is_full)
        self.assertEqual(len(window), 0)

    def test_analyzer_runs_only_for_full_window(self) -> None:
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
        self.assertIsNone(analyze_window(window, analyzer))

        window.append(
            KafkaEvent.from_dict(
                {"event_id": 2, "event_time": now.isoformat(), "value": 2}
            )
        )
        report = analyze_window(window, analyzer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["value"].tolist(), [1, 2])
        self.assertEqual(report["overall_status"], "ok")


    def test_analysis_counter_matches_non_overlapping_window_readiness_formula(
        self,
    ) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.set_window_size(3)

        for _ in range(7):
            exporter.record_processed_event()
        for _ in range(2):
            exporter.record_analysis_run()

        metrics = generate_latest(registry).decode()
        self.assertIn("drift_events_processed_total 7.0", metrics)
        self.assertIn("drift_analysis_runs_total 2.0", metrics)
        # формула readiness: 7 - 2 * 3 = 1 событие в текущем окне
        self.assertEqual(7 - 2 * 3, 1)

    def test_stream_quality_is_window_local_and_worst_status_wins(self) -> None:
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
        self.assertEqual(warmup.status, -1)

        snapshot = tracker.snapshot(window.event_times(), ready=True)
        self.assertAlmostEqual(snapshot.invalid_event_time_rate, 1 / 3)
        self.assertEqual(snapshot.status, 1)

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
        self.assertEqual(snapshot.invalid_event_time_rate, 0.0)
        self.assertEqual(snapshot.status, 0)

    def test_stream_status_is_insufficient_without_configured_thresholds(
        self,
    ) -> None:
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

        self.assertEqual(snapshot.status, -1)


    def test_stream_status_uses_cumulative_late_events_total(self) -> None:
        tracker = StreamTracker(
            late_event_threshold_seconds=1,
            thresholds=StreamThresholds(
                event_time_lag_seconds=ThresholdPair(100, 200),
                late_events_total=ThresholdPair(1, 2),
            ),
        )
        now = datetime.now(UTC)

        first = KafkaEvent.from_dict(
            {
                "event_id": 1,
                "event_time": (now - timedelta(seconds=5)).isoformat(),
                "value": 1,
            }
        )
        tracker.observe(first)
        self.assertEqual(tracker.snapshot([first.event_time], ready=False).status, -1)
        self.assertEqual(tracker.snapshot([first.event_time], ready=True).status, 1)

        second = KafkaEvent.from_dict(
            {
                "event_id": 2,
                "event_time": (now - timedelta(seconds=4)).isoformat(),
                "value": 2,
            }
        )
        tracker.observe(second)
        self.assertEqual(
            tracker.snapshot([first.event_time, second.event_time], ready=True).status,
            2,
        )

    def test_lifetime_counters_survive_window_reset(self) -> None:
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
        self.assertEqual(before.late_events_total, 2)
        self.assertEqual(before.out_of_order_events_total, 1)

        tracker.reset_window()
        after = tracker.snapshot([], ready=True)
        self.assertEqual(after.late_events_total, 2)
        self.assertEqual(after.out_of_order_events_total, 1)
        self.assertEqual(after.invalid_event_time_rate, 0.0)

    def test_reference_profile_exports_only_agreed_metadata(self) -> None:
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
        self.assertIn(
            'drift_reference_profile_info{dataset_name="demo",'
            'profile_created_at="2026-09-16T10:00:00Z"} 1.0',
            metrics,
        )
        self.assertNotIn("profile_name", metrics)
        self.assertNotIn("profile_version", metrics)
        self.assertIn("drift_reference_sample_size 10000.0", metrics)

    def test_exporter_supports_yulia_generic_contract(
        self,
    ) -> None:
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
        self.assertIn(
            'drift_status_feature{feature="age",type="numeric"} 2.0', metrics
        )
        self.assertIn(
            'drift_metric_value{feature="age",metric="psi",type="numeric"} 0.31',
            metrics,
        )
        self.assertIn(
            'drift_status{feature="age",metric="psi",type="numeric"} 2.0',
            metrics,
        )
        self.assertIn(
            'drift_threshold{level="warning",metric="psi"} 0.1', metrics
        )
        self.assertIn(
            'drift_metric_value{feature="prediction",'
            'metric="prediction_psi",type="numeric"} 0.14',
            metrics,
        )
        self.assertIn(
            'drift_threshold{level="warning",metric="prediction_psi"} 0.1',
            metrics,
        )
        self.assertIn("drift_analysis_runs_total 1.0", metrics)
        self.assertIn(
            'drift_stream_threshold{level="warning",'
            'metric="drift_invalid_event_time_rate"} 0.05',
            metrics,
        )
        self.assertIn(
            'drift_stream_threshold{level="critical",'
            'metric="drift_max_event_gap_seconds"} 30.0',
            metrics,
        )
        self.assertRegex(
            metrics,
            r"drift_report_timestamp_seconds [1-9][0-9]*(?:\.[0-9]+)?(?:e\+[0-9]+)?",
        )


    def test_metric_status_falls_back_to_feature_status_and_prediction_resets(
        self,
    ) -> None:
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
        self.assertIn(
            'drift_status_feature{feature="age",type="numeric"} 1.0',
            metrics,
        )
        self.assertIn(
            'drift_status_feature{feature="prediction",type="numeric"} 1.0',
            metrics,
        )

        exporter.update_report(
            {
                "overall_status": "ok",
                "features": {},
            }
        )
        metrics = generate_latest(registry).decode()
        self.assertNotIn('feature="prediction"', metrics)

    def test_exporter_accepts_flat_report_from_current_yulia_core(self) -> None:
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
        self.assertIn(
            'drift_metric_value{feature="age",metric="psi",type="numeric"} 0.31',
            metrics,
        )
        self.assertIn(
            'drift_status{feature="age",metric="psi",type="numeric"} 2.0',
            metrics,
        )
        self.assertIn(
            'drift_threshold{level="critical",metric="psi"} 0.25', metrics
        )


if __name__ == "__main__":
    unittest.main()
