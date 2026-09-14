import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.engine import analyze_dataframe
from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.engine_adapter import (
    EngineAdapter,
    build_engine_adapter_from_env,
)
from drift_guardian.ingestion.window import WindowBuffer, analyze_window
from drift_guardian.ingestion.event import KafkaEvent


class EngineAdapterTest(unittest.TestCase):
    def test_adapter_calls_two_dataframe_core_api(self) -> None:
        reference_df = pd.DataFrame({"value": [10, 20, 30]})
        current_df = pd.DataFrame({"value": [1, 2]})
        calls = []

        def core(reference, current):
            calls.append((reference.copy(), current.copy()))
            return {"overall_status": "ok", "features": {}}

        adapter = EngineAdapter(
            reference_df=reference_df,
            dataset_name="unit_reference",
            profile_created_at="2026-09-14T12:00:00Z",
            core_analyzer=core,
        )
        report = adapter(current_df)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["value"].tolist(), [10, 20, 30])
        self.assertEqual(calls[0][1]["value"].tolist(), [1, 2])
        self.assertEqual(report["window_size"], 2)
        self.assertIn("timestamp", report)
        self.assertEqual(adapter.get_reference_metadata()["sample_size"], 3)

    def test_build_adapter_from_env_samples_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            pd.DataFrame({"value": range(50)}).to_csv(path, index=False)

            env = {
                "REFERENCE_DATA_PATH": str(path),
                "REFERENCE_SAMPLE_MULTIPLIER": "3",
                "REFERENCE_SAMPLE_RANDOM_STATE": "7",
            }
            with patch.dict(os.environ, env, clear=False):
                adapter = build_engine_adapter_from_env(window_size=4)

        metadata = adapter.get_reference_metadata()
        self.assertEqual(len(adapter.reference_df), 12)
        self.assertEqual(metadata["dataset_name"], "reference")
        self.assertEqual(metadata["sample_size"], 12)

    def test_current_core_to_realtime_to_prometheus_integration(self) -> None:
        adapter = EngineAdapter(
            reference_df=pd.DataFrame(
                {
                    "age": [30, 40, 50],
                    "income": [50_000, 60_000, 70_000],
                    "country": ["DE", "FR", "EE"],
                    "prediction_score": [0.2, 0.5, 0.8],
                }
            ),
            dataset_name="integration_reference",
            profile_created_at="2026-09-14T12:00:00Z",
            core_analyzer=analyze_dataframe,
        )

        window = WindowBuffer(2)
        window.append(
            KafkaEvent.from_dict(
                {
                    "event_id": 1,
                    "event_time": "2026-09-14T12:00:00Z",
                    "age": 35,
                    "income": 55_000,
                    "country": "DE",
                    "prediction_score": 0.3,
                }
            )
        )
        window.append(
            KafkaEvent.from_dict(
                {
                    "event_id": 2,
                    "event_time": "2026-09-14T12:00:01Z",
                    "age": 45,
                    "income": 65_000,
                    "country": "FR",
                    "prediction_score": 0.6,
                }
            )
        )

        report = analyze_window(window, adapter, min_window_size=2)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["window_size"], 2)
        self.assertEqual(report["overall_status"], "critical")

        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.update_reference_profile(adapter.get_reference_metadata())
        exporter.update_report(report)
        exporter.record_analysis_run()

        metrics = generate_latest(registry).decode()
        self.assertIn("drift_analysis_runs_total 1.0", metrics)
        self.assertIn("drift_window_size 2.0", metrics)
        self.assertIn(
            'drift_feature_psi{feature="age",type="numeric"} 0.31',
            metrics,
        )
        self.assertIn(
            'drift_reference_profile_info{dataset_name="integration_reference",profile_created_at="2026-09-14T12:00:00Z"} 1.0',
            metrics,
        )
        self.assertIn("drift_reference_sample_size 3.0", metrics)
        self.assertRegex(
            metrics,
            r"drift_report_timestamp_seconds [1-9][0-9]*(?:\\.[0-9]+)?",
        )


if __name__ == "__main__":
    unittest.main()
