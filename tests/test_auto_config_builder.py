from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import yaml

import config.auto_config_builder as acb
from config.parse_config import config_from_dict


@pytest.fixture(autouse=True)
def patch_metric_registry(monkeypatch):
    """
    Изолируем тесты auto_config_builder от реальных реализаций метрик.

    auto_config_builder проверяет наличие метрик в METRIC_REGISTRY и вызывает
    функции метрик при auto_thresholds.enabled=True.
    """

    def make_metric_fn(metric_name: str):
        def fn(reference_dict, current: pd.Series, **kwargs):
            if metric_name == "missing_rate":
                return float(current.isna().mean())

            # chi2 в проекте трактуется как reversed metric/p-value:
            # меньше — хуже. Для тестов возвращаем стабильный p-value.
            if metric_name == "chi2":
                return 1.0

            # Остальные drift-метрики в тестах считаем нулевыми,
            # чтобы проверить floors.
            return 0.0

        return fn

    metric_names = [
        "missing_rate",
        "psi",
        "unseen_category_rate",
        "cardinality_ratio",
        "js_divergence",
        "wasserstein_distance",
        "chi2",
        "cramer_v",
        "category_churn",
        "kstest",
    ]

    registry = {
        acb._metric(name): make_metric_fn(name)
        for name in metric_names
    }

    monkeypatch.setattr(acb, "METRIC_REGISTRY", registry)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27],
            "country": ["RU", "RU", "US", "KZ", "RU", "US", "AM", "RU"],
            "score": [0.1, 0.2, 0.3, 0.35, 0.5, 0.6, 0.7, 0.9],
            "event_time": pd.date_range("2026-01-01", periods=8),
            "all_missing": [np.nan] * 8,
        }
    )


def test_build_basic_config_filters_columns_and_is_valid_by_schema(df):
    options = acb.ConfigBuildOptions(
        prediction_score_column="score",
        numeric_metrics=["missing_rate", "psi"],
        categorical_metrics=["missing_rate", "psi", "unseen_category_rate"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
            "psi": (0.1, 0.25),
            "unseen_category_rate": [0.01, 0.05],
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    config = acb.build_drift_config(df, options)

    assert config["prediction_metrics"] == {"enabled": False}

    # prediction_score_column исключается из обычных features.
    assert set(config["features"]) == {"age", "country"}

    assert config["features"]["age"] == {
        "type": "numeric",
        "metrics": ["missing_rate", "psi"],
    }

    assert config["features"]["country"] == {
        "type": "categorical",
        "metrics": ["missing_rate", "psi", "unseen_category_rate"],
    }

    # Должен успешно пройти pydantic-валидацию основного Config.
    parsed = config_from_dict(config)

    assert parsed.features["age"].type == "numeric"
    assert parsed.features["country"].type == "categorical"


def test_build_writes_yaml_file(df, tmp_path):
    output_path = tmp_path / "drift_config.yaml"

    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        numeric_metrics=["missing_rate"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    config = acb.build_drift_config(
        df=df,
        options=options,
        output_path=str(output_path),
    )

    assert output_path.exists()

    with output_path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert loaded == config
    assert loaded["features"]["age"]["type"] == "numeric"


def test_prediction_block_numeric_default_metrics_and_schema_validation(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        prediction_enabled=True,
        prediction_score_column="score",
        numeric_metrics=["missing_rate"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
            "psi": {"warning": 0.1, "critical": 0.25},
            "kstest": {"warning": 0.02, "critical": 0.05},
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    config = acb.build_drift_config(df, options)

    assert config["prediction_metrics"] == {
        "enabled": True,
        "score_column": "score",
        "type": "numeric",
        "metrics": ["psi", "kstest"],
    }

    # Проверяем, что итоговый dict совместим со схемой Config.
    config_from_dict(config)


def test_prediction_requires_score_column(df):
    options = acb.ConfigBuildOptions(
        prediction_enabled=True,
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    with pytest.raises(
        ValueError,
        match="prediction_enabled=True requires prediction_score_column",
    ):
        acb.build_drift_config(df, options)


def test_include_columns_missing_raises(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age", "unknown_column"],
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    with pytest.raises(ValueError, match="include_columns are missing in df"):
        acb.build_drift_config(df, options)


def test_time_column_raises_when_drop_time_columns_false(df):
    options = acb.ConfigBuildOptions(
        include_columns=["event_time"],
        drop_time_columns=False,
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    with pytest.raises(ValueError, match="has time dtype"):
        acb.build_drift_config(df, options)


def test_all_missing_column_raises_when_drop_all_missing_false(df):
    options = acb.ConfigBuildOptions(
        include_columns=["all_missing"],
        drop_all_missing_columns=False,
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    with pytest.raises(ValueError, match="contains only missing values"):
        acb.build_drift_config(df, options)


def test_strict_incompatible_metric_raises(df):
    options = acb.ConfigBuildOptions(
        include_columns=["country"],
        categorical_metrics=["kstest"],
        global_thresholds={
            "kstest": {"warning": 0.02, "critical": 0.05},
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
        strict_metric_compatibility=True,
    )

    with pytest.raises(
        ValueError,
        match="Metric 'kstest' is incompatible with feature 'country' type 'categorical'",
    ):
        acb.build_drift_config(df, options)


def test_non_strict_incompatible_metric_warns_and_drops(df):
    options = acb.ConfigBuildOptions(
        include_columns=["country"],
        categorical_metrics=["kstest", "missing_rate"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
            "kstest": {"warning": 0.02, "critical": 0.05},
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
        strict_metric_compatibility=False,
    )

    with pytest.warns(UserWarning, match="Dropping metric"):
        config = acb.build_drift_config(df, options)

    assert config["features"]["country"]["metrics"] == ["missing_rate"]


def test_disabled_metrics_are_removed(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        numeric_metrics=["missing_rate", "psi"],
        disabled_metrics=["psi"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
            "psi": {"warning": 0.1, "critical": 0.25},
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    config = acb.build_drift_config(df, options)

    assert config["features"]["age"]["metrics"] == ["missing_rate"]


def test_feature_local_thresholds_are_normalized(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        numeric_metrics=["missing_rate", "psi"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
            "psi": {"warning": 0.1, "critical": 0.25},
        },
        feature_thresholds={
            "age": {
                "psi": [0.2, 0.4],
            }
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    config = acb.build_drift_config(df, options)

    assert config["features"]["age"]["thresholds"] == {
        "psi": {
            "warning": 0.2,
            "critical": 0.4,
        }
    }

    parsed = config_from_dict(config)
    assert parsed.features["age"].resolved_thresholds[acb.MetricEnum("psi")].warning == 0.2


def test_missing_thresholds_raise_when_auto_thresholds_disabled(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        numeric_metrics=["missing_rate", "psi"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
            # psi threshold отсутствует намеренно
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    with pytest.raises(
        ValueError,
        match="Missing thresholds after auto-generation",
    ):
        acb.build_drift_config(df, options)


def test_stream_drift_thresholds_are_added_and_normalized(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        numeric_metrics=["missing_rate"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
        },
        stream_drift={
            "drift_event_time_lag_seconds": [10, 30],
            "drift_invalid_event_time_rate": {"warning": 0.01, "critical": 0.05},
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    config = acb.build_drift_config(df, options)

    assert config["stream_drift"] == {
        "drift_event_time_lag_seconds": {
            "warning": 10.0,
            "critical": 30.0,
        },
        "drift_invalid_event_time_rate": {
            "warning": 0.01,
            "critical": 0.05,
        },
    }

    config_from_dict(config)


def test_invalid_stream_drift_direction_raises(df):
    options = acb.ConfigBuildOptions(
        include_columns=["age"],
        numeric_metrics=["missing_rate"],
        global_thresholds={
            "missing_rate": {"warning": 0.01, "critical": 0.05},
        },
        stream_drift={
            "drift_event_time_lag_seconds": [30, 10],
        },
        auto_thresholds=acb.AutoThresholdSettings(enabled=False),
    )

    with pytest.raises(ValueError, match="warning must be less than critical"):
        acb.build_drift_config(df, options)


def test_auto_thresholds_generate_global_feature_and_prediction_thresholds(df, monkeypatch):
    class DummyProfiler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def profile_ref_data(self):
            return {
                "cat_ref": {},
                "num_ref": {},
                "sample": pd.DataFrame(),
                "preds_ref": {},
            }

    monkeypatch.setattr(acb, "Profiler", DummyProfiler)

    options = acb.ConfigBuildOptions(
        include_columns=["age", "country"],
        prediction_enabled=True,
        prediction_score_column="score",
        prediction_metrics=["missing_rate"],
        numeric_metrics=["missing_rate"],
        categorical_metrics=["missing_rate"],
        profiler_window_size=4,
        auto_thresholds=acb.AutoThresholdSettings(
            enabled=True,
            method="bootstrap",
            window_size=4,
            n_windows=3,
            per_feature=True,
            per_prediction=True,
            random_state=123,
        ),
    )

    config = acb.build_drift_config(df, options)

    expected_floor = {
        "warning": 0.005,
        "critical": 0.02,
    }

    assert config["thresholds"]["missing_rate"] == expected_floor

    assert config["features"]["age"]["thresholds"]["missing_rate"] == expected_floor
    assert config["features"]["country"]["thresholds"]["missing_rate"] == expected_floor
    assert config["prediction_metrics"]["thresholds"]["missing_rate"] == expected_floor

    config_from_dict(config)


def test_threshold_from_values_uses_floor_for_regular_metric():
    options = acb.ConfigBuildOptions(
        auto_thresholds=acb.AutoThresholdSettings(enabled=True)
    )

    result = acb._threshold_from_values(
        metric_name="psi",
        values=[0.0, 0.0, 0.0],
        options=options,
    )

    assert result == {
        "warning": 0.1,
        "critical": 0.25,
    }


def test_threshold_from_values_uses_cap_for_reversed_metric():
    options = acb.ConfigBuildOptions(
        auto_thresholds=acb.AutoThresholdSettings(enabled=True)
    )

    result = acb._threshold_from_values(
        metric_name="chi2",
        values=[1.0, 1.0, 1.0],
        options=options,
    )

    assert result == {
        "warning": 0.05,
        "critical": 0.01,
    }


def test_threshold_direction_validation_regular_metric():
    assert acb._normalize_metric_threshold_pair("psi", [0.1, 0.25]) == {
        "warning": 0.1,
        "critical": 0.25,
    }

    with pytest.raises(
        ValueError,
        match="For metric 'psi' warning must be less than critical",
    ):
        acb._normalize_metric_threshold_pair("psi", [0.25, 0.1])


def test_threshold_direction_validation_reversed_metric():
    assert acb._normalize_metric_threshold_pair("chi2", [0.05, 0.01]) == {
        "warning": 0.05,
        "critical": 0.01,
    }

    with pytest.raises(
        ValueError,
        match="For reversed metric 'chi2' warning must be greater than critical",
    ):
        acb._normalize_metric_threshold_pair("chi2", [0.01, 0.05])


@pytest.mark.parametrize(
    "auto_settings, error_match",
    [
        (
            acb.AutoThresholdSettings(window_size=0),
            "auto_thresholds.window_size must be positive",
        ),
        (
            acb.AutoThresholdSettings(n_windows=0),
            "auto_thresholds.n_windows must be positive",
        ),
        (
            acb.AutoThresholdSettings(warning_quantile=0),
            "warning_quantile must be in",
        ),
        (
            acb.AutoThresholdSettings(critical_quantile=1),
            "critical_quantile must be in",
        ),
        (
            acb.AutoThresholdSettings(warning_quantile=0.99, critical_quantile=0.95),
            "warning_quantile must be less than critical_quantile",
        ),
        (
            acb.AutoThresholdSettings(eps=0),
            "eps must be positive",
        ),
    ],
)
def test_validate_auto_settings_errors(auto_settings, error_match):
    with pytest.raises(ValueError, match=error_match):
        acb._validate_auto_settings(auto_settings)


def test_iter_calibration_windows_bootstrap_is_deterministic(df):
    windows_1 = list(
        acb._iter_calibration_windows(
            df=df,
            method="bootstrap",
            window_size=4,
            n_windows=3,
            random_state=123,
        )
    )

    windows_2 = list(
        acb._iter_calibration_windows(
            df=df,
            method="bootstrap",
            window_size=4,
            n_windows=3,
            random_state=123,
        )
    )

    assert len(windows_1) == 3
    assert all(len(window) == 4 for window in windows_1)

    for left, right in zip(windows_1, windows_2):
        pd.testing.assert_frame_equal(left, right)


def test_iter_calibration_windows_rolling(df):
    windows = list(
        acb._iter_calibration_windows(
            df=df,
            method="rolling",
            window_size=4,
            n_windows=3,
            random_state=123,
        )
    )

    assert len(windows) == 3
    assert [list(window.index) for window in windows] == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
    ]


def test_iter_calibration_windows_unknown_method_raises(df):
    with pytest.raises(ValueError, match="Unknown calibration method"):
        list(
            acb._iter_calibration_windows(
                df=df,
                method="unknown",  # type: ignore[arg-type]
                window_size=4,
                n_windows=3,
                random_state=123,
            )
        )


def test_compute_metric_safe_returns_none_for_non_finite(monkeypatch, df):
    metric = acb._metric("psi")

    def bad_metric(reference_dict, current, **kwargs):
        return math.inf

    monkeypatch.setitem(acb.METRIC_REGISTRY, metric, bad_metric)

    options = acb.ConfigBuildOptions()

    result = acb._compute_metric_safe(
        metric_name="psi",
        reference_dict={},
        current=df["age"],
        options=options,
        label="feature 'age'",
    )

    assert result is None


def test_compute_metric_safe_warns_and_returns_none_when_ignore_errors_true(monkeypatch, df):
    metric = acb._metric("psi")

    def failing_metric(reference_dict, current, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(acb.METRIC_REGISTRY, metric, failing_metric)

    options = acb.ConfigBuildOptions(
        auto_thresholds=acb.AutoThresholdSettings(ignore_metric_errors=True)
    )

    with pytest.warns(UserWarning, match="Metric 'psi' failed"):
        result = acb._compute_metric_safe(
            metric_name="psi",
            reference_dict={},
            current=df["age"],
            options=options,
            label="feature 'age'",
        )

    assert result is None


def test_compute_metric_safe_raises_when_ignore_errors_false(monkeypatch, df):
    metric = acb._metric("psi")

    def failing_metric(reference_dict, current, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(acb.METRIC_REGISTRY, metric, failing_metric)

    options = acb.ConfigBuildOptions(
        auto_thresholds=acb.AutoThresholdSettings(ignore_metric_errors=False)
    )

    with pytest.raises(RuntimeError, match="Metric 'psi' failed"):
        acb._compute_metric_safe(
            metric_name="psi",
            reference_dict={},
            current=df["age"],
            options=options,
            label="feature 'age'",
        )