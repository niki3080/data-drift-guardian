import unittest
from datetime import UTC, datetime, timedelta

from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.stream_metrics import StreamTracker
from drift_guardian.ingestion.window import WindowBuffer, analyze_window


class IngestionTest(unittest.TestCase):
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

    def test_stream_metrics(self) -> None:
        tracker = StreamTracker(late_event_threshold_seconds=1)
        window = WindowBuffer(3)
        now = datetime.now(UTC)

        tracker.record_invalid_event_time()
        for seconds in (5, 10):
            event = KafkaEvent.from_dict(
                {
                    "event_id": seconds,
                    "event_time": (now - timedelta(seconds=seconds)).isoformat(),
                    "value": seconds,
                }
            )
            window.append(event)
            tracker.observe(event)

        snapshot = tracker.snapshot(window.event_times(), len(window), 2)

        self.assertEqual(snapshot.status, 0)
        self.assertEqual(snapshot.window_size, 2)
        self.assertEqual(snapshot.late_events_total, 2)
        self.assertEqual(snapshot.out_of_order_events_total, 1)
        self.assertAlmostEqual(snapshot.invalid_event_time_rate, 1 / 3)

    def test_prometheus_report_contract(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.update_report(
            {
                "timestamp": "2026-09-13T11:32:00Z",
                "window_size": 1000,
                "overall_status": "critical",
                "active_alerts": 3,
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
                            "missing_rate": {
                                "value": 0.01,
                                "warning": 0.05,
                                "critical": 0.1,
                                "status": "ok",
                            },
                        },
                    }
                },
                "prediction": {
                    "status": "critical",
                    "metrics": {
                        "prediction_psi": {
                            "value": 0.29,
                            "warning": 0.1,
                            "critical": 0.25,
                            "status": "critical",
                        }
                    },
                },
            }
        )

        metrics = generate_latest(registry).decode()

        self.assertIn("drift_overall_status 2.0", metrics)
        self.assertIn("drift_active_alerts 3.0", metrics)
        self.assertIn(
            'drift_metric_value{feature="age",metric="psi"} 0.31',
            metrics,
        )
        self.assertIn(
            'drift_threshold{feature="age",level="critical",metric="psi"} 0.25',
            metrics,
        )
        self.assertIn(
            'drift_status{feature="age",metric="psi"} 2.0',
            metrics,
        )
        self.assertIn(
            'drift_metric_value{feature="prediction",metric="prediction_psi"} 0.29',
            metrics,
        )
        self.assertNotIn("_created", metrics)


if __name__ == "__main__":
    unittest.main()
