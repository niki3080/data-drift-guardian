import re
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

import drift_guardian.analyzer.engine.engine as engine_module
from drift_guardian.analyzer.engine.engine import DriftMetricsEngine


@dataclass(frozen=True)
class DummyMetric:
    value: str


@dataclass
class DummyThreshold:
    warning: float
    critical: float


def make_feature_config(metrics=None, thresholds=None):
    return SimpleNamespace(
        metrics=metrics or [],
        resolved_thresholds=thresholds or {},
    )


def make_prediction_config(
    enabled=False,
    score_column="prediction_score",
    metrics=None,
    thresholds=None,
):
    return SimpleNamespace(
        enabled=enabled,
        score_column=score_column,
        metrics=metrics or [],
        resolved_thresholds=thresholds or {},
    )


def make_config(features=None, prediction_metrics=None):
    return SimpleNamespace(
        features=features or {},
        prediction_metrics=prediction_metrics or make_prediction_config(enabled=False),
    )


def test_engine_init_stores_config_and_reference_profile():
    config = make_config()
    reference_profile = {
        "sample": pd.DataFrame({"x": [1, 2, 3]})
    }

    engine = DriftMetricsEngine(config, reference_profile)

    assert engine.config is config
    assert engine.reference_dict is reference_profile
    assert engine.window_size is None


@pytest.mark.parametrize(
    ("warning", "critical", "value", "expected"),
    [
        # lower is better: warning < critical
        (0.2, 0.5, 0.1, "ok"),
        (0.2, 0.5, 0.3, "warning"),
        (0.2, 0.5, 0.6, "critical"),

        # higher is better: warning > critical
        (0.8, 0.5, 0.9, "ok"),
        (0.8, 0.5, 0.7, "warning"),
        (0.8, 0.5, 0.4, "critical"),
    ],
)
def test_get_metric_status(warning, critical, value, expected):
    assert DriftMetricsEngine._get_metric_status(
        warning,
        critical,
        value,
    ) == expected


def test_get_feature_status_returns_ok_when_all_metrics_ok():
    metrics_result = {
        "metric_1": {"status": "ok"},
        "metric_2": {"status": "ok"},
    }

    assert DriftMetricsEngine._get_feature_status(metrics_result) == "ok"


def test_get_feature_status_returns_warning_when_at_least_one_metric_warning():
    metrics_result = {
        "metric_1": {"status": "ok"},
        "metric_2": {"status": "warning"},
    }

    assert DriftMetricsEngine._get_feature_status(metrics_result) == "warning"


def test_get_feature_status_returns_critical_when_non_chi2_metric_is_critical():
    metrics_result = {
        "metric_1": {"status": "warning"},
        "metric_2": {"status": "critical"},
    }

    assert DriftMetricsEngine._get_feature_status(metrics_result) == "critical"


def test_get_feature_status_ignores_critical_chi2_metric():
    metrics_result = {
        engine_module.Metric.chi2: {"status": "critical"},
    }

    assert DriftMetricsEngine._get_feature_status(metrics_result) == "ok"


def test_get_feature_status_returns_warning_when_chi2_critical_and_other_metric_warning():
    metrics_result = {
        engine_module.Metric.chi2: {"status": "critical"},
        "some_metric": {"status": "warning"},
    }

    assert DriftMetricsEngine._get_feature_status(metrics_result) == "warning"


def test_get_overall_status_returns_ok_when_all_features_ok():
    monitoring_results = {
        "age": {"status": "ok"},
        "income": {"status": "ok"},
    }

    overall_status, active_alerts = DriftMetricsEngine._get_overall_status(
        monitoring_results
    )

    assert overall_status == "ok"
    assert active_alerts == 0


def test_get_overall_status_returns_warning_when_warning_exists_without_critical():
    monitoring_results = {
        "age": {"status": "ok"},
        "income": {"status": "warning"},
    }

    overall_status, active_alerts = DriftMetricsEngine._get_overall_status(
        monitoring_results
    )

    assert overall_status == "warning"
    assert active_alerts == 0


def test_get_overall_status_returns_critical_and_counts_critical_features():
    monitoring_results = {
        "age": {"status": "critical"},
        "income": {"status": "warning"},
        "city": {"status": "critical"},
    }

    overall_status, active_alerts = DriftMetricsEngine._get_overall_status(
        monitoring_results
    )

    assert overall_status == "critical"
    assert active_alerts == 2


def test_get_current_timestamp_returns_utc_iso_string_with_z_suffix():
    timestamp = DriftMetricsEngine._get_current_timestamp()

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        timestamp,
    )


def test_make_report_without_prediction(monkeypatch):
    config = make_config()
    reference_profile = {
        "sample": pd.DataFrame()
    }

    engine = DriftMetricsEngine(config, reference_profile)
    engine.window_size = 100

    monkeypatch.setattr(
        DriftMetricsEngine,
        "_get_current_timestamp",
        staticmethod(lambda: "2026-09-19T05:36:00Z"),
    )

    features_result = {
        "age": {
            "status": "ok",
            "metrics": {},
        },
        "income": {
            "status": "warning",
            "metrics": {},
        },
    }

    report = engine._make_report(
        features_result,
        monitoring_prediction_result=None,
    )

    assert report == {
        "timestamp": "2026-09-19T05:36:00Z",
        "window_size": 100,
        "overall_status": "warning",
        "active_alerts": 0,
        "features": features_result,
    }

    assert "prediction" not in report


def test_make_report_with_prediction(monkeypatch):
    config = make_config()
    reference_profile = {
        "sample": pd.DataFrame()
    }

    engine = DriftMetricsEngine(config, reference_profile)
    engine.window_size = 50

    monkeypatch.setattr(
        DriftMetricsEngine,
        "_get_current_timestamp",
        staticmethod(lambda: "2026-09-19T05:36:00Z"),
    )

    features_result = {
        "age": {
            "status": "critical",
            "metrics": {},
        },
    }

    prediction_result = {
        "type": "prediction",
        "status": "ok",
        "metrics": {},
    }

    report = engine._make_report(
        features_result,
        monitoring_prediction_result=prediction_result,
    )

    assert report == {
        "timestamp": "2026-09-19T05:36:00Z",
        "window_size": 50,
        "overall_status": "critical",
        "active_alerts": 1,
        "features": features_result,
        "prediction": prediction_result,
    }


def test_analyze_column_for_feature(monkeypatch):
    metric_warning = DummyMetric("dummy_warning_metric")
    metric_critical = DummyMetric("dummy_critical_metric")

    calls = []

    def warning_metric_fn(reference_dict, current):
        calls.append(("warning", reference_dict, current))
        return 0.3

    def critical_metric_fn(reference_dict, current):
        calls.append(("critical", reference_dict, current))
        return 0.8

    # Кладём ключи и по объекту metric, и по metric.value,
    # чтобы тест был устойчивым к реализации:
    # METRIC_REGISTRY[metric] или METRIC_REGISTRY[metric.value].
    monkeypatch.setattr(
        engine_module,
        "METRIC_REGISTRY",
        {
            metric_warning: warning_metric_fn,
            metric_warning.value: warning_metric_fn,
            metric_critical: critical_metric_fn,
            metric_critical.value: critical_metric_fn,
        },
    )

    monkeypatch.setattr(
        engine_module,
        "find_ref",
        lambda reference_dict, column_name: {"type": "numeric"},
    )

    thresholds = {
        metric_warning: DummyThreshold(warning=0.2, critical=0.5),
        metric_warning.value: DummyThreshold(warning=0.2, critical=0.5),
        metric_critical: DummyThreshold(warning=0.2, critical=0.5),
        metric_critical.value: DummyThreshold(warning=0.2, critical=0.5),
    }

    config = make_config(
        features={
            "age": make_feature_config(
                metrics=[metric_warning, metric_critical],
                thresholds=thresholds,
            )
        }
    )

    reference_profile = {
        "sample": pd.DataFrame({"age": [10, 20, 30]})
    }

    engine = DriftMetricsEngine(config, reference_profile)

    current_column = pd.Series([11, 22, 33], name="age")

    result = engine._analyze_column(
        reference_profile,
        current_column,
        config,
        column_type="feature",
    )

    assert result == {
        "type": "numeric",
        "status": "critical",
        "metrics": {
            "dummy_warning_metric": {
                "value": 0.3,
                "warning": 0.2,
                "critical": 0.5,
                "status": "warning",
            },
            "dummy_critical_metric": {
                "value": 0.8,
                "warning": 0.2,
                "critical": 0.5,
                "status": "critical",
            },
        },
    }

    assert len(calls) == 2

    assert calls[0][1] is reference_profile
    assert calls[0][2] is current_column

    assert calls[1][1] is reference_profile
    assert calls[1][2] is current_column


def test_analyze_column_for_prediction(monkeypatch):
    prediction_metric = DummyMetric("prediction_metric")

    def prediction_metric_fn(reference_dict, current):
        return 0.1

    monkeypatch.setattr(
        engine_module,
        "METRIC_REGISTRY",
        {
            prediction_metric: prediction_metric_fn,
            prediction_metric.value: prediction_metric_fn,
        },
    )

    monkeypatch.setattr(
        engine_module,
        "find_ref",
        lambda reference_dict, column_name: {"type": "prediction"},
    )

    thresholds = {
        prediction_metric: DummyThreshold(warning=0.2, critical=0.5),
        prediction_metric.value: DummyThreshold(warning=0.2, critical=0.5),
    }

    config = make_config(
        features={},
        prediction_metrics=make_prediction_config(
            enabled=True,
            score_column="score",
            metrics=[prediction_metric],
            thresholds=thresholds,
        ),
    )

    reference_profile = {
        "sample": pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    }

    engine = DriftMetricsEngine(config, reference_profile)

    current_column = pd.Series([0.15, 0.25, 0.35], name="score")

    result = engine._analyze_column(
        reference_profile,
        current_column,
        config,
        column_type="prediction",
    )

    assert result == {
        "type": "prediction",
        "status": "ok",
        "metrics": {
            "prediction_metric": {
                "value": 0.1,
                "warning": 0.2,
                "critical": 0.5,
                "status": "ok",
            },
        },
    }


def test_analyze_column_raises_value_error_for_unknown_column_type(monkeypatch):
    metric = DummyMetric("some_metric")

    monkeypatch.setattr(
        engine_module,
        "find_ref",
        lambda reference_dict, column_name: {"type": "numeric"},
    )

    config = make_config(
        features={
            "age": make_feature_config(
                metrics=[metric],
                thresholds={
                    metric: DummyThreshold(warning=0.2, critical=0.5),
                    metric.value: DummyThreshold(warning=0.2, critical=0.5),
                },
            )
        }
    )

    reference_profile = {
        "sample": pd.DataFrame({"age": [1, 2, 3]})
    }

    engine = DriftMetricsEngine(config, reference_profile)

    current_column = pd.Series([1, 2, 3], name="age")

    with pytest.raises(
        ValueError,
        match="column_type must be feature or prediction",
    ):
        engine._analyze_column(
            engine.reference_dict,
            current_column,
            config,
            column_type="unknown",
        )


def test_analyze_dataframe_with_features_and_prediction(monkeypatch):
    feature_config = make_feature_config(metrics=[], thresholds={})

    config = make_config(
        features={
            "age": feature_config,
            "income": feature_config,
        },
        prediction_metrics=make_prediction_config(
            enabled=True,
            score_column="score",
        ),
    )

    reference_profile = {
        "sample": pd.DataFrame()
    }

    engine = DriftMetricsEngine(config, reference_profile)

    current = pd.DataFrame(
        {
            "age": [10, 20, 30],
            "income": [100, 200, 300],
            "score": [0.1, 0.2, 0.3],
        }
    )

    calls = []

    def fake_analyze_column(
        self,
        reference_dict,
        column_current,
        config_arg,
        column_type="feature",
    ):
        calls.append(
            {
                "reference_dict": reference_dict,
                "column_name": column_current.name,
                "config": config_arg,
                "column_type": column_type,
            }
        )

        return {
            "type": column_type,
            "status": "ok",
            "metrics": {},
        }

    monkeypatch.setattr(
        DriftMetricsEngine,
        "_analyze_column",
        fake_analyze_column,
    )

    monkeypatch.setattr(
        DriftMetricsEngine,
        "_get_current_timestamp",
        staticmethod(lambda: "2026-09-19T05:36:00Z"),
    )

    report = engine.analyze_dataframe(current)

    assert engine.window_size == 3

    assert calls == [
        {
            "reference_dict": reference_profile,
            "column_name": "age",
            "config": config,
            "column_type": "feature",
        },
        {
            "reference_dict": reference_profile,
            "column_name": "income",
            "config": config,
            "column_type": "feature",
        },
        {
            "reference_dict": reference_profile,
            "column_name": "score",
            "config": config,
            "column_type": "prediction",
        },
    ]

    assert report == {
        "timestamp": "2026-09-19T05:36:00Z",
        "window_size": 3,
        "overall_status": "ok",
        "active_alerts": 0,
        "features": {
            "age": {
                "type": "feature",
                "status": "ok",
                "metrics": {},
            },
            "income": {
                "type": "feature",
                "status": "ok",
                "metrics": {},
            },
        },
        "prediction": {
            "type": "prediction",
            "status": "ok",
            "metrics": {},
        },
    }


def test_analyze_dataframe_without_prediction(monkeypatch):
    feature_config = make_feature_config(metrics=[], thresholds={})

    config = make_config(
        features={
            "age": feature_config,
        },
        prediction_metrics=make_prediction_config(enabled=False),
    )

    reference_profile = {
        "sample": pd.DataFrame()
    }

    engine = DriftMetricsEngine(config, reference_profile)

    current = pd.DataFrame(
        {
            "age": [10, 20, 30],
            "score": [0.1, 0.2, 0.3],
        }
    )

    calls = []

    def fake_analyze_column(
        self,
        reference_dict,
        column_current,
        config_arg,
        column_type="feature",
    ):
        calls.append(
            {
                "reference_dict": reference_dict,
                "column_name": column_current.name,
                "config": config_arg,
                "column_type": column_type,
            }
        )

        return {
            "type": column_type,
            "status": "ok",
            "metrics": {},
        }

    monkeypatch.setattr(
        DriftMetricsEngine,
        "_analyze_column",
        fake_analyze_column,
    )

    monkeypatch.setattr(
        DriftMetricsEngine,
        "_get_current_timestamp",
        staticmethod(lambda: "2026-09-19T05:36:00Z"),
    )

    report = engine.analyze_dataframe(current)

    assert engine.window_size == 3

    assert calls == [
        {
            "reference_dict": reference_profile,
            "column_name": "age",
            "config": config,
            "column_type": "feature",
        },
    ]

    assert report == {
        "timestamp": "2026-09-19T05:36:00Z",
        "window_size": 3,
        "overall_status": "ok",
        "active_alerts": 0,
        "features": {
            "age": {
                "type": "feature",
                "status": "ok",
                "metrics": {},
            },
        },
    }

    assert "prediction" not in report


def test_run_adversarial_validation_calls_function_with_expected_arguments(monkeypatch):
    reference_sample = pd.DataFrame(
        {
            "age": [10, 20, 30],
            "city": ["A", "B", "C"],
        }
    )

    current = pd.DataFrame(
        {
            "age": [11, 22, 33],
            "city": ["A", "B", "D"],
        }
    )

    expected_auc = 0.75

    expected_importance = pd.DataFrame(
        {
            "feature": ["age", "city"],
            "importance": [0.8, 0.2],
            "importance_std": [0.1, 0.05],
            "rank": [1, 2],
        }
    )

    calls = {}

    def fake_adversarial_validation(
        reference,
        current_arg,
        *args,
        **kwargs,
    ):
        calls["reference"] = reference
        calls["current"] = current_arg

        param_names = [
            "max_samples",
            "n_splits",
            "random_state",
            "missing_category",
            "lightgbm_params",
        ]

        for index, name in enumerate(param_names):
            if index < len(args):
                calls[name] = args[index]
            else:
                calls[name] = kwargs.get(name)

        return expected_auc, expected_importance

    monkeypatch.setattr(
        engine_module,
        "adversarial_validation",
        fake_adversarial_validation,
    )

    config = make_config()

    reference_profile = {
        "sample": reference_sample
    }

    engine = DriftMetricsEngine(config, reference_profile)

    result_auc, result_importance = engine.run_adversarial_validation(
        current,
        max_samples=10,
        n_splits=2,
        random_state=123,
        missing_category="MISSING",
        lightgbm_params={"n_estimators": 10},
    )

    assert result_auc == expected_auc
    pd.testing.assert_frame_equal(result_importance, expected_importance)

    assert calls["reference"] is reference_sample
    assert calls["current"] is current
    assert calls["max_samples"] == 10
    assert calls["n_splits"] == 2
    assert calls["random_state"] == 123
    assert calls["missing_category"] == "MISSING"
    assert calls["lightgbm_params"] == {"n_estimators": 10}

