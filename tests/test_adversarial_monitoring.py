from __future__ import annotations

import unittest

import pandas as pd
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.engine_adapter import EngineAdapter, _av_status


class AdversarialMonitoringTests(unittest.TestCase):
    def test_av_status_threshold_boundaries_are_inclusive(self) -> None:
        thresholds = {"warning": 0.60, "critical": 0.75}
        self.assertEqual(_av_status(0.59, thresholds), "ok")
        self.assertEqual(_av_status(0.60, thresholds), "warning")
        self.assertEqual(_av_status(0.74, thresholds), "warning")
        self.assertEqual(_av_status(0.75, thresholds), "critical")

    def test_engine_adapter_adds_av_status_from_configured_thresholds(self) -> None:
        reference = pd.DataFrame(
            {
                "age": [30.0, 40.0, 50.0],
                "country": ["DE", "FR", "GB"],
            }
        )
        current = pd.DataFrame(
            {
                "age": [35.0, 45.0, 55.0],
                "country": ["DE", "ES", "GB"],
            }
        )

        def core_analyzer(reference_df, current_df):
            return {"overall_status": "ok", "features": {}}

        def fake_av(reference_df, current_df):
            importance = pd.DataFrame(
                {
                    "feature": ["age", "country"],
                    "importance": [0.7, 0.3],
                    "importance_std": [0.1, 0.1],
                    "rank": [1, 2],
                }
            )
            importance.attrs["roc_auc_cv_mean"] = 0.78
            importance.attrs["roc_auc_cv_std"] = 0.012
            importance.attrs["roc_auc_cv_min"] = 0.765
            importance.attrs["roc_auc_cv_max"] = 0.795
            importance.attrs["driver_consistency"] = 0.91
            return 0.78, importance

        adapter = EngineAdapter(
            reference_df=reference,
            dataset_name="demo",
            profile_created_at="2026-09-20T12:00:00Z",
            core_analyzer=core_analyzer,
            adversarial_analyzer=fake_av,
            adversarial_top_features=1,
            adversarial_thresholds={"warning": 0.60, "critical": 0.75},
        )

        report = adapter(current)
        av = report["adversarial_validation"]

        self.assertEqual(av["roc_auc"], 0.78)
        self.assertEqual(av["roc_auc_cv_mean"], 0.78)
        self.assertEqual(av["roc_auc_cv_std"], 0.012)
        self.assertEqual(av["roc_auc_cv_min"], 0.765)
        self.assertEqual(av["roc_auc_cv_max"], 0.795)
        self.assertEqual(av["driver_consistency"], 0.91)
        self.assertEqual(av["top1_importance_share"], 0.7)
        self.assertEqual(av["top3_importance_share"], 1.0)
        self.assertEqual(av["reference_rows"], 3)
        self.assertEqual(av["current_rows"], 3)
        self.assertEqual(av["dataset_size"], 3)
        self.assertEqual(av["features_evaluated"], 2)
        self.assertEqual(av["reference_sample_fraction"], 1.0)
        self.assertEqual(av["current_sample_fraction"], 1.0)
        self.assertEqual(av["status"], "critical")
        self.assertEqual(av["thresholds"], {"warning": 0.60, "critical": 0.75})
        self.assertEqual(
            av["feature_importance"],
            [{"feature": "age", "importance": 0.7, "rank": 1}],
        )


    def test_engine_adapter_tracks_driver_similarity_between_windows(self) -> None:
        reference = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        current = pd.DataFrame({"a": [4.0, 5.0, 6.0], "b": [4.0, 5.0, 6.0]})
        calls = iter(([0.8, 0.2], [0.2, 0.8]))

        def fake_av(_reference_df, _current_df):
            values = next(calls)
            return 0.7, pd.DataFrame(
                {
                    "feature": ["a", "b"],
                    "importance": values,
                    "importance_std": [0.0, 0.0],
                    "rank": [1, 2],
                }
            )

        adapter = EngineAdapter(
            reference_df=reference,
            dataset_name="demo",
            profile_created_at="2026-09-20T12:00:00Z",
            core_analyzer=lambda _reference, _current: {
                "overall_status": "ok",
                "features": {},
            },
            adversarial_analyzer=fake_av,
        )

        first = adapter(current)["adversarial_validation"]
        second = adapter(current)["adversarial_validation"]
        self.assertNotIn("driver_similarity_previous", first)
        self.assertAlmostEqual(
            second["driver_similarity_previous"],
            0.47058823529411764,
        )

    def test_engine_adapter_does_not_invent_av_status_without_thresholds(self) -> None:
        reference = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        current = pd.DataFrame({"value": [4.0, 5.0, 6.0]})

        def fake_av(_reference_df, _current_df):
            return 0.8, pd.DataFrame(
                {
                    "feature": ["value"],
                    "importance": [1.0],
                    "importance_std": [0.0],
                    "rank": [1],
                }
            )

        adapter = EngineAdapter(
            reference_df=reference,
            dataset_name="demo",
            profile_created_at="2026-09-20T12:00:00Z",
            core_analyzer=lambda _reference, _current: {
                "overall_status": "ok",
                "features": {},
            },
            adversarial_analyzer=fake_av,
        )

        av = adapter(current)["adversarial_validation"]
        self.assertNotIn("status", av)
        self.assertEqual(av["dataset_size"], 3)

    def test_prometheus_exporter_exposes_av_contract(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.update_report(
            {
                "timestamp": "2026-09-20T12:00:00Z",
                "overall_status": "ok",
                "features": {},
                "adversarial_validation": {
                    "timestamp": "2026-09-20T12:00:00Z",
                    "status": "warning",
                    "roc_auc": 0.78,
                    "roc_auc_cv_mean": 0.78,
                    "roc_auc_cv_std": 0.012,
                    "roc_auc_cv_min": 0.765,
                    "roc_auc_cv_max": 0.795,
                    "driver_consistency": 0.91,
                    "driver_similarity_previous": 0.84,
                    "top1_importance_share": 0.29,
                    "top3_importance_share": 0.64,
                    "reference_rows": 10_000,
                    "current_rows": 1_000,
                    "dataset_size": 1_000,
                    "features_evaluated": 20,
                    "reference_sample_fraction": 0.1,
                    "current_sample_fraction": 1.0,
                    "thresholds": {"warning": 0.60, "critical": 0.75},
                    "feature_importance": [
                        {"feature": "age", "importance": 0.29, "rank": 1},
                        {"feature": "income", "importance": 0.21, "rank": 2},
                    ],
                },
            }
        )

        metrics = generate_latest(registry).decode("utf-8")
        self.assertIn("drift_av_available 1.0", metrics)
        self.assertIn("drift_av_status 1.0", metrics)
        self.assertIn("drift_av_roc_auc 0.78", metrics)
        self.assertIn("drift_av_roc_auc_cv_mean 0.78", metrics)
        self.assertIn("drift_av_roc_auc_cv_std 0.012", metrics)
        self.assertIn("drift_av_roc_auc_cv_min 0.765", metrics)
        self.assertIn("drift_av_roc_auc_cv_max 0.795", metrics)
        self.assertIn("drift_av_driver_consistency 0.91", metrics)
        self.assertIn("drift_av_driver_similarity_previous 0.84", metrics)
        self.assertIn("drift_av_top1_importance_share 0.29", metrics)
        self.assertIn("drift_av_top3_importance_share 0.64", metrics)
        self.assertIn("drift_av_dataset_size 1000.0", metrics)
        self.assertIn(
            "drift_av_last_run_timestamp_seconds 1.7899056e+09",
            metrics,
        )
        self.assertIn("drift_av_reference_rows 10000.0", metrics)
        self.assertIn("drift_av_current_rows 1000.0", metrics)
        self.assertIn("drift_av_features_evaluated 20.0", metrics)
        self.assertIn('drift_av_sample_fraction{dataset="reference"} 0.1', metrics)
        self.assertIn('drift_av_sample_fraction{dataset="current"} 1.0', metrics)
        self.assertIn('drift_av_threshold{level="warning"} 0.6', metrics)
        self.assertIn('drift_av_threshold{level="critical"} 0.75', metrics)
        self.assertIn(
            'drift_av_feature_importance{feature="age",rank="1"} 0.29',
            metrics,
        )

    def test_missing_av_block_resets_av_metrics(self) -> None:
        registry = CollectorRegistry()
        exporter = PrometheusExporter(registry)
        exporter.update_adversarial_validation(
            {
                "status": "critical",
                "roc_auc": 0.9,
                "feature_importance": [
                    {"feature": "age", "importance": 1.0, "rank": 1}
                ],
            }
        )
        exporter.update_adversarial_validation(None)

        metrics = generate_latest(registry).decode("utf-8")
        self.assertIn("drift_av_available 0.0", metrics)
        self.assertIn("drift_av_status -1.0", metrics)
        self.assertIn("drift_av_roc_auc -1.0", metrics)
        self.assertIn("drift_av_roc_auc_cv_mean -1.0", metrics)
        self.assertIn("drift_av_roc_auc_cv_std -1.0", metrics)
        self.assertIn("drift_av_roc_auc_cv_min -1.0", metrics)
        self.assertIn("drift_av_roc_auc_cv_max -1.0", metrics)
        self.assertIn("drift_av_driver_consistency -1.0", metrics)
        self.assertIn("drift_av_driver_similarity_previous -1.0", metrics)
        self.assertIn("drift_av_top1_importance_share -1.0", metrics)
        self.assertIn("drift_av_top3_importance_share -1.0", metrics)
        self.assertIn("drift_av_dataset_size 0.0", metrics)
        self.assertIn("drift_av_features_evaluated 0.0", metrics)
        self.assertNotIn("drift_av_sample_fraction{", metrics)
        self.assertNotIn("drift_av_feature_importance{", metrics)
        self.assertNotIn("drift_av_threshold{", metrics)


if __name__ == "__main__":
    unittest.main()
