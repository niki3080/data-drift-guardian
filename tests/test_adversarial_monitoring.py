from __future__ import annotations

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.engine_adapter import EngineAdapter, _av_status


@pytest.mark.parametrize(
    ("roc_auc", "expected_status"),
    [
        (0.59, "ok"),
        (0.60, "warning"),
        (0.74, "warning"),
        (0.75, "critical"),
    ],
)
def test_av_status_threshold_boundaries_are_inclusive(
    roc_auc: float,
    expected_status: str,
) -> None:
    thresholds = {"warning": 0.60, "critical": 0.75}

    assert _av_status(roc_auc, thresholds) == expected_status


def test_engine_adapter_adds_av_status_from_configured_thresholds() -> None:
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

    def core_analyzer(_reference_df, _current_df):
        return {"overall_status": "ok", "features": {}}

    def fake_av(_reference_df, _current_df):
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

    av = adapter(current)["adversarial_validation"]

    assert av["roc_auc"] == 0.78
    assert av["roc_auc_cv_mean"] == 0.78
    assert av["roc_auc_cv_std"] == 0.012
    assert av["roc_auc_cv_min"] == 0.765
    assert av["roc_auc_cv_max"] == 0.795
    assert av["driver_consistency"] == 0.91
    assert av["top1_importance_share"] == 0.7
    assert av["top3_importance_share"] == 1.0
    assert av["reference_rows"] == 3
    assert av["current_rows"] == 3
    assert av["dataset_size"] == 3
    assert av["features_evaluated"] == 2
    assert av["reference_sample_fraction"] == 1.0
    assert av["current_sample_fraction"] == 1.0
    assert av["status"] == "critical"
    assert av["thresholds"] == {"warning": 0.60, "critical": 0.75}
    assert av["feature_importance"] == [
        {"feature": "age", "importance": 0.7, "rank": 1}
    ]


def test_engine_adapter_tracks_driver_similarity_between_windows() -> None:
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

    assert "driver_similarity_previous" not in first
    assert second["driver_similarity_previous"] == pytest.approx(
        0.47058823529411764
    )


def test_engine_adapter_does_not_invent_av_status_without_thresholds() -> None:
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

    assert "status" not in av
    assert av["dataset_size"] == 3


def test_prometheus_exporter_exposes_av_contract() -> None:
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

    for sample in (
        "drift_av_available 1.0",
        "drift_av_status 1.0",
        "drift_av_roc_auc 0.78",
        "drift_av_roc_auc_cv_mean 0.78",
        "drift_av_roc_auc_cv_std 0.012",
        "drift_av_roc_auc_cv_min 0.765",
        "drift_av_roc_auc_cv_max 0.795",
        "drift_av_driver_consistency 0.91",
        "drift_av_driver_similarity_previous 0.84",
        "drift_av_top1_importance_share 0.29",
        "drift_av_top3_importance_share 0.64",
        "drift_av_dataset_size 1000.0",
        "drift_av_last_run_timestamp_seconds 1.7899056e+09",
        "drift_av_reference_rows 10000.0",
        "drift_av_current_rows 1000.0",
        "drift_av_features_evaluated 20.0",
        'drift_av_sample_fraction{dataset="reference"} 0.1',
        'drift_av_sample_fraction{dataset="current"} 1.0',
        'drift_av_threshold{level="warning"} 0.6',
        'drift_av_threshold{level="critical"} 0.75',
        'drift_av_feature_importance{feature="age",rank="1"} 0.29',
    ):
        assert sample in metrics


def test_missing_av_block_resets_av_metrics() -> None:
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

    for sample in (
        "drift_av_available 0.0",
        "drift_av_status -1.0",
        "drift_av_roc_auc -1.0",
        "drift_av_roc_auc_cv_mean -1.0",
        "drift_av_roc_auc_cv_std -1.0",
        "drift_av_roc_auc_cv_min -1.0",
        "drift_av_roc_auc_cv_max -1.0",
        "drift_av_driver_consistency -1.0",
        "drift_av_driver_similarity_previous -1.0",
        "drift_av_top1_importance_share -1.0",
        "drift_av_top3_importance_share -1.0",
        "drift_av_dataset_size 0.0",
        "drift_av_features_evaluated 0.0",
    ):
        assert sample in metrics

    assert "drift_av_sample_fraction{" not in metrics
    assert "drift_av_feature_importance{" not in metrics
    assert "drift_av_threshold{" not in metrics
