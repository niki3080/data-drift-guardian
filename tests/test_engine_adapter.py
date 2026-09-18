import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.engine import analyze_dataframe
from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.demo_reference import build_demo_reference
from drift_guardian.ingestion.engine_adapter import (
    EngineAdapter,
    build_engine_adapter_from_env,
)
from drift_guardian.ingestion.event import KafkaEvent
from drift_guardian.ingestion.window import WindowBuffer, analyze_window


CONFIG = """
features:
  age:
    type: numeric
    metrics: [missing_rate, psi]
  income:
    type: numeric
    metrics: [missing_rate, psi]
  country:
    type: categorical
    metrics: [missing_rate, psi, unseen_category_rate, cardinality_ratio]
prediction_metrics:
  enabled: true
  score_column: prediction_score
  type: numeric
  metrics: [psi]
thresholds:
  psi:
    warning: 0.1
    critical: 0.25
  unseen_category_rate:
    warning: 0.05
    critical: 0.1
"""


class EngineAdapterTest(unittest.TestCase):
    def test_demo_reference_is_reproducible(self) -> None:
        first = build_demo_reference(rows=5, seed=42)
        second = build_demo_reference(rows=5, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_adapter_calls_current_yulia_two_dataframe_core_api(self) -> None:
        reference_df = pd.DataFrame({"value": [10, 20, 30]})
        current_df = pd.DataFrame({"value": [1, 2]})
        calls = []

        def core(reference, current):
            calls.append((reference.copy(), current.copy()))
            return {
                "timestamp": "2026-09-16T10:00:00Z",
                "overall_status": "ok",
                "features": {},
            }

        adapter = EngineAdapter(
            reference_df=reference_df,
            dataset_name="unit_reference",
            profile_created_at="2026-09-16T10:00:00Z",
            core_analyzer=core,
        )
        report = adapter(current_df)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["value"].tolist(), [10, 20, 30])
        self.assertEqual(calls[0][1]["value"].tolist(), [1, 2])
        self.assertEqual(report["window_size"], 2)
        # timestamp Core сохраняем в читаемом ISO-8601 формате
        self.assertEqual(report["timestamp"], "2026-09-16T10:00:00Z")

    def test_schema_checker_from_lev_validates_against_reference(self) -> None:
        reference = pd.DataFrame(
            {
                "age": pd.Series([30, 40], dtype="int64"),
                "income": pd.Series([50_000.0, 60_000.0], dtype="float64"),
                "country": pd.Series(["DE", "FR"], dtype="object"),
                "prediction_score": pd.Series([0.2, 0.8], dtype="float64"),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            config = Path(directory) / "config.yaml"
            reference.to_csv(path, index=False)
            config.write_text(CONFIG, encoding="utf-8")

            adapter = EngineAdapter.from_path(
                path,
                sample_size=2,
                config_path=config,
            )
            validated = adapter.validate_features(
                {
                    "age": 35,
                    "income": 55_000.0,
                    "country": "DE",
                    "prediction_score": 0.3,
                    "extra": "dropped",
                }
            )
            self.assertEqual(set(validated), set(reference.columns))

            # отсутствующие фичи сохраняем как None, чтобы Core мог считать
            # missing_rate и не отбрасывать всё Kafka-событие
            missing_prediction = adapter.validate_features(
                {
                    "age": 35,
                    "income": 55_000.0,
                    "country": "DE",
                }
            )
            self.assertIsNone(missing_prediction["prediction_score"])

            with self.assertRaises(ValueError):
                adapter.validate_features(
                    {
                        "age": "not-an-int",
                        "income": 55_000.0,
                        "country": "DE",
                        "prediction_score": 0.3,
                    }
                )


    def test_adapter_merges_config_and_report_thresholds(self) -> None:
        reference_df = pd.DataFrame({"value": [1, 2, 3]})

        def core(_reference, _current):
            return {
                "overall_status": "ok",
                "features": {},
                "thresholds": {
                    "psi": {"critical": 0.3},
                    "kstest": {"warning": 0.1, "critical": 0.2},
                },
            }

        adapter = EngineAdapter(
            reference_df=reference_df,
            dataset_name="threshold_reference",
            profile_created_at="2026-09-16T10:00:00Z",
            core_analyzer=core,
            thresholds={
                "psi": {"warning": 0.1, "critical": 0.25},
                "missing_rate": {"warning": 0.05, "critical": 0.1},
            },
        )

        report = adapter(pd.DataFrame({"value": [4]}))
        self.assertEqual(
            report["thresholds"]["psi"],
            {"warning": 0.1, "critical": 0.3},
        )
        self.assertEqual(
            report["thresholds"]["missing_rate"],
            {"warning": 0.05, "critical": 0.1},
        )
        self.assertEqual(
            report["thresholds"]["kstest"],
            {"warning": 0.1, "critical": 0.2},
        )

    def test_build_adapter_from_env_generates_demo_reference_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.csv"
            config = Path(directory) / "config.yaml"
            config.write_text(CONFIG, encoding="utf-8")
            env = {
                "REFERENCE_DATA_PATH": str(path),
                "GENERATE_DEMO_REFERENCE": "true",
                "DEMO_REFERENCE_ROWS": "50",
                "DEMO_REFERENCE_SEED": "7",
                "REFERENCE_SAMPLE_MULTIPLIER": "3",
                "REFERENCE_SAMPLE_RANDOM_STATE": "7",
                "DRIFT_CONFIG_PATH": str(config),
            }
            with patch.dict(os.environ, env, clear=False):
                adapter = build_engine_adapter_from_env(window_size=4)

            self.assertTrue(path.exists())
            self.assertEqual(len(adapter.reference_df), 12)
            self.assertEqual(adapter.get_reference_metadata()["sample_size"], 12)
            self.assertIn("psi", adapter.thresholds)

    def test_current_yulia_core_to_realtime_to_prometheus_integration(self) -> None:
        reference = build_demo_reference(rows=10, seed=42)
        adapter = EngineAdapter(
            reference_df=reference,
            dataset_name="integration_reference",
            profile_created_at="2026-09-16T10:00:00Z",
            core_analyzer=analyze_dataframe,
            thresholds={"psi": {"warning": 0.1, "critical": 0.25}},
            prediction_type="numeric",
        )

        window = WindowBuffer(2)
        for event_id, age in ((1, 35), (2, 45)):
            window.append(
                KafkaEvent.from_dict(
                    {
                        "event_id": event_id,
                        "event_time": f"2026-09-16T10:00:0{event_id}Z",
                        "age": age,
                        "income": 55_000.0 + event_id,
                        "country": "DE",
                        "prediction_score": 0.3 + event_id / 10,
                    }
                )
            )

        report = analyze_window(window, adapter)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["window_size"], 2)
        self.assertIn("timestamp", report)
        self.assertIn("thresholds", report)

        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.set_window_size(2)
        exporter.update_reference_profile(adapter.get_reference_metadata())
        exporter.update_report(report)
        exporter.record_analysis_run()

        metrics = generate_latest(registry).decode()
        self.assertIn("drift_analysis_runs_total 1.0", metrics)
        self.assertIn(
            'drift_metric_value{feature="age",metric="psi",type="numeric"} 0.31',
            metrics,
        )
        self.assertIn(
            'drift_threshold{level="warning",metric="psi"} 0.1', metrics
        )
        self.assertIn("drift_reference_sample_size 10.0", metrics)

        # Временный JSON-stub должен соответствовать актуальному config contract.
        self.assertEqual(
            set(report["features"]["age"]["metrics"]),
            {"missing_rate", "psi"},
        )
        self.assertEqual(
            set(report["features"]["income"]["metrics"]),
            {"missing_rate", "psi"},
        )
        self.assertEqual(
            set(report["features"]["country"]["metrics"]),
            {
                "missing_rate",
                "psi",
                "unseen_category_rate",
                "cardinality_ratio",
            },
        )
        self.assertEqual(set(report["prediction"]["metrics"]), {"psi"})
        self.assertNotIn("mean_zscore", metrics)
        self.assertNotIn("positive_prediction_rate", metrics)
        self.assertIn(
            'drift_metric_value{feature="prediction",'
            'metric="prediction_psi",type="numeric"} 0.11',
            metrics,
        )


if __name__ == "__main__":
    unittest.main()
