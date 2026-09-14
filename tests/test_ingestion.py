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

        def factory(config):
            configs.append(config)
            return fake_admin

        wait_for_kafka_topic(
            "kafka:19092",
            "features-stream",
            timeout_seconds=5.0,
            retry_interval_seconds=0.5,
            admin_client_factory=factory,
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
        self.assertEqual(event.features["prediction_score"], 0.72)

        with self.assertRaises(InvalidEventTime):
            KafkaEvent.from_dict(
                {
                    "event_id": 2,
                    "event_time": "2026-09-13T11:32:00",
                    "age": 45,
                }
            )

    def test_sliding_window(self) -> None:
        window = WindowBuffer(3)
        now = datetime.now(UTC)

        for event_id in range(1, 5):
            window.append(
                KafkaEvent.from_dict(
                    {
                        "event_id": event_id,
                        "event_time": now.isoformat(),
                        "value": event_id,
                    }
                )
            )

        self.assertEqual(len(window), 3)
        self.assertEqual(window.to_dataframe()["value"].tolist(), [2, 3, 4])

    def test_window_is_dataframe_above_analyzer(self) -> None:
        calls = []
        window = WindowBuffer(3)
        now = datetime.now(UTC)

        window.append(
            KafkaEvent.from_dict(
                {
                    "event_id": 1,
                    "event_time": now.isoformat(),
                    "value": 1,
                }
            )
        )

        def analyzer(current_df):
            calls.append(current_df.copy())
            return {"overall_status": "ok", "features": {}}

        self.assertIsNone(analyze_window(window, analyzer, 2))

        window.append(
            KafkaEvent.from_dict(
                {
                    "event_id": 2,
                    "event_time": now.isoformat(),
                    "value": 2,
                }
            )
        )
        report = analyze_window(window, analyzer, 2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["value"].tolist(), [1, 2])
        self.assertEqual(report["overall_status"], "ok")

    def test_stream_quality_is_window_local_and_worst_status_wins(self) -> None:
        thresholds = StreamThresholds(
            invalid_event_time_rate=ThresholdPair(0.2, 0.5),
            late_event_rate=ThresholdPair(0.2, 0.5),
            out_of_order_event_rate=ThresholdPair(0.2, 0.5),
        )
        tracker = StreamTracker(
            window_size=3,
            late_event_threshold_seconds=1000,
            thresholds=thresholds,
        )
        window = WindowBuffer(3)
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

        snapshot = tracker.snapshot(window.event_times(), len(window), 2)
        self.assertAlmostEqual(snapshot.invalid_event_time_rate, 1 / 3)
        self.assertEqual(snapshot.status, 1)

        event = KafkaEvent.from_dict(
            {
                "event_id": 3,
                "event_time": (now + timedelta(seconds=3)).isoformat(),
                "value": 3,
            }
        )
        window.append(event)
        tracker.observe(event)
        snapshot = tracker.snapshot(window.event_times(), len(window), 2)

        self.assertEqual(snapshot.invalid_event_time_rate, 0.0)
        self.assertEqual(snapshot.status, 0)

    def test_lifetime_counters_are_separate_from_window_rates(self) -> None:
        tracker = StreamTracker(window_size=2, late_event_threshold_seconds=1)
        window = WindowBuffer(2)
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
        for event in (first, second):
            window.append(event)
            tracker.observe(event)

        snapshot = tracker.snapshot(window.event_times(), len(window), 2)
        self.assertEqual(snapshot.late_events_total, 2)
        self.assertEqual(snapshot.out_of_order_events_total, 1)
        self.assertEqual(snapshot.late_event_rate, 1.0)
        self.assertEqual(snapshot.out_of_order_event_rate, 0.5)

    def test_exporter_accepts_actual_main_report_shape(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.update_report(
            {
                "overall_status": "critical",
                "active_alerts": 2,
                "window_size": 1000,
                "thresholds": {
                    "psi": {"warning": 0.1, "critical": 0.25},
                },
                "features": {
                    "age": {
                        "type": "numeric",
                        "status": "critical",
                        "metrics": {
                            "psi": 0.31,
                            "missing_rate": 0.01,
                            "mean_zscore": 3.4,
                        },
                        "alerts": ["psi_critical", "mean_zscore_critical"],
                    },
                    "income": {
                        "type": "numeric",
                        "status": "passed",
                        "metrics": {"psi": 0.04},
                        "alerts": [],
                    },
                },
                "prediction": {
                    "status": "passed",
                    "metrics": {
                        "prediction_score_drift": 0.08,
                        "positive_prediction_rate": 0.41,
                    },
                    "alerts": [],
                },
            }
        )
        exporter.record_analysis_run()
        exporter.update_reference_profile(
            {
                "profile_created_at": "2026-09-13T10:00:00Z",
                "dataset_name": "transactions",
                "sample_size": 10000,
            }
        )

        metrics = generate_latest(registry).decode()
        self.assertIn(
            'drift_feature_psi{feature="age",type="numeric"} 0.31',
            metrics,
        )
        self.assertIn(
            'drift_feature_status{feature="income",type="numeric"} 0.0',
            metrics,
        )
        self.assertIn(
            'drift_feature_alert_info{alert="psi_critical",feature="age",type="numeric"} 1.0',
            metrics,
        )
        self.assertIn(
            'drift_metric_threshold{level="critical",metric="psi"} 0.25',
            metrics,
        )
        self.assertIn("drift_analysis_runs_total 1.0", metrics)
        self.assertIn("drift_prediction_score_drift 0.08", metrics)
        self.assertIn("drift_prediction_positive_rate 0.41", metrics)
        self.assertIn(
            'drift_reference_profile_info{dataset_name="transactions",profile_created_at="2026-09-13T10:00:00Z"} 1.0',
            metrics,
        )
        self.assertIn("drift_reference_sample_size 10000.0", metrics)
        self.assertNotIn("drift_analysis_runs_created", metrics)
        self.assertNotIn("drift_events_processed_created", metrics)

    def test_exporter_still_accepts_nested_metric_payloads(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.update_report(
            {
                "overall_status": "warning",
                "active_alerts": 1,
                "window_size": 100,
                "features": {
                    "age": {
                        "type": "numeric",
                        "status": "warning",
                        "metrics": {
                            "psi": {
                                "value": 0.15,
                                "warning": 0.1,
                                "critical": 0.25,
                                "status": "warning",
                            }
                        },
                    }
                },
            }
        )

        metrics = generate_latest(registry).decode()
        self.assertIn(
            'drift_feature_psi{feature="age",type="numeric"} 0.15',
            metrics,
        )
        self.assertIn(
            'drift_feature_alert_info{alert="psi_warning",feature="age",type="numeric"} 1.0',
            metrics,
        )

    def test_stream_thresholds_are_exported_only_when_configured(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.set_stream_thresholds(StreamThresholds())
        metrics = generate_latest(registry).decode()
        self.assertNotIn("drift_stream_threshold{", metrics)

        exporter.set_stream_thresholds(
            StreamThresholds(
                max_event_gap_seconds=ThresholdPair(5.0, 15.0),
            )
        )
        metrics = generate_latest(registry).decode()
        self.assertIn(
            'drift_stream_threshold{level="warning",metric="max_event_gap_seconds"} 5.0',
            metrics,
        )
        self.assertIn(
            'drift_stream_threshold{level="critical",metric="max_event_gap_seconds"} 15.0',
            metrics,
        )


if __name__ == "__main__":
    unittest.main()
