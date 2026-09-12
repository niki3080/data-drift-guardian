import unittest
from datetime import UTC, datetime, timedelta

from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.realtime.event import InvalidEventTime, KafkaEvent
from drift_guardian.realtime.realtime_monitor import RealTimeDriftMonitor
from drift_guardian.realtime.window_buffer import WindowBuffer


class RealtimeTest(unittest.TestCase):
    def test_event(self) -> None:
        event = KafkaEvent.from_dict(
            {
                "event_id": 1,
                "event_time": "2026-08-16T10:42:00Z",
                "age": 45,
                "prediction_score": 0.72,
            }
        )

        self.assertEqual(event.event_id, 1)
        self.assertEqual(event.features["age"], 45)
        self.assertEqual(event.features["prediction_score"], 0.72)

        with self.assertRaises(InvalidEventTime):
            KafkaEvent.from_dict(
                {
                    "event_id": 2,
                    "event_time": "2026-08-16T10:42:00",
                    "age": 45,
                }
            )

    def test_window(self) -> None:
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

    def test_monitor_calls_engine(self) -> None:
        calls = []

        class Engine:
            def analyze_dataframe(self, current_df):
                calls.append(current_df.copy())
                return {
                    "overall_status": "ok",
                    "active_alerts": 0,
                    "features": {},
                }

        monitor = RealTimeDriftMonitor(
            engine=Engine(),
            window_size=3,
            min_window_size=2,
        )
        now = datetime.now(UTC)

        first = KafkaEvent.from_dict(
            {
                "event_id": 1,
                "event_time": now.isoformat(),
                "value": 1,
            }
        )
        second = KafkaEvent.from_dict(
            {
                "event_id": 2,
                "event_time": (now + timedelta(seconds=1)).isoformat(),
                "value": 2,
            }
        )

        self.assertIsNone(monitor.update(first))
        report = monitor.update(second)

        self.assertEqual(len(calls), 1)
        self.assertEqual(report["window_size"], 2)

    def test_stream_metrics(self) -> None:
        monitor = RealTimeDriftMonitor(
            window_size=3,
            min_window_size=2,
            late_event_threshold_seconds=1,
        )
        now = datetime.now(UTC)

        monitor.record_invalid_event_time()
        monitor.update(
            KafkaEvent.from_dict(
                {
                    "event_id": 1,
                    "event_time": (now - timedelta(seconds=5)).isoformat(),
                    "value": 1,
                }
            )
        )
        monitor.update(
            KafkaEvent.from_dict(
                {
                    "event_id": 2,
                    "event_time": (now - timedelta(seconds=10)).isoformat(),
                    "value": 2,
                }
            )
        )

        snapshot = monitor.stream_snapshot()
        self.assertEqual(snapshot.status, 0)
        self.assertEqual(snapshot.window_size, 2)
        self.assertEqual(snapshot.late_events_total, 2)
        self.assertEqual(snapshot.out_of_order_events_total, 1)
        self.assertAlmostEqual(snapshot.invalid_event_time_rate, 1 / 3)

    def test_prometheus_exporter(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.record_processed_event()

        monitor = RealTimeDriftMonitor(window_size=3, min_window_size=1)
        monitor.update(
            KafkaEvent.from_dict(
                {
                    "event_id": 1,
                    "event_time": datetime.now(UTC).isoformat(),
                    "value": 1,
                }
            )
        )
        exporter.update_stream(monitor.stream_snapshot())
        exporter.update_report(
            {
                "overall_status": "warning",
                "active_alerts": 1,
                "window_size": 1,
                "features": {
                    "value": {
                        "type": "numeric",
                        "status": "warning",
                        "metrics": {"psi": 0.2},
                    }
                },
                "prediction": {
                    "status": "ok",
                    "metrics": {
                        "prediction_psi": 0.01,
                        "positive_prediction_rate": 0.5,
                    },
                },
            }
        )

        metrics = generate_latest(registry).decode()

        self.assertIn("drift_events_processed_total 1.0", metrics)
        self.assertIn("drift_window_size 1.0", metrics)
        self.assertIn("drift_overall_status 1.0", metrics)
        self.assertIn(
            'drift_feature_psi{feature="value",type="numeric"} 0.2',
            metrics,
        )
        self.assertIn("drift_prediction_positive_rate 0.5", metrics)
        self.assertNotIn("_created", metrics)


if __name__ == "__main__":
    unittest.main()
