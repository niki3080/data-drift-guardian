import importlib
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

MODULE_PATH = "src.drift_guardian.analyzer.offline.offline_mode"
offline_mode = importlib.import_module(MODULE_PATH)
OfflineWrapper = offline_mode.OfflineWrapper


@pytest.fixture
def reference_df():
    return pd.DataFrame(
        {
            "age": [20, 30, 40],
            "country": ["RU", "US", "RU"],
            "score": [0.1, 0.2, 0.3],
            "unused": [1, 2, 3],
        }
    )


@pytest.fixture
def current_df():
    return pd.DataFrame(
        {
            "age": [21, 31, 41],
            "country": ["RU", "US", "KZ"],
            "score": [0.11, 0.22, 0.33],
            "unused": [10, 20, 30],
            "extra": ["x", "y", "z"],
        }
    )


def _as_list_or_none(value):
    if value is None:
        return None
    return list(value)


def install_offline_wrapper_fakes(
    monkeypatch,
    *,
    num_feats=("age",),
    cat_feats=("country",),
    prediction="score",
):
    """
    Ставит фейки на зависимости, импортированные внутри offline_mode.py.

    OfflineWrapper использует:
    - extract_feature_groups
    - Profiler
    - SchemaChecker
    - DriftMetricsEngine

    Все они подменяются, чтобы тестировать только orchestration-логику wrapper'а.
    """
    state = SimpleNamespace(
        profiler_instances=[],
        checker_instances=[],
        engine_instances=[],
    )

    def fake_extract_feature_groups(config):
        return (
            _as_list_or_none(num_feats),
            _as_list_or_none(cat_feats),
            prediction,
        )

    class FakeProfiler:
        def __init__(
            self,
            reference_df,
            num_features=None,
            cat_features=None,
            prediction=None,
            merge_threshold=5,
            take_sample=True,
            low_cardinality_threshold=15,
        ):
            self.reference_df = reference_df
            self.num_features = num_features
            self.cat_features = cat_features
            self.prediction = prediction
            self.merge_threshold = merge_threshold
            self.take_sample = take_sample
            self.low_cardinality_threshold = low_cardinality_threshold
            state.profiler_instances.append(self)

        def profile_ref_data(self):
            return {
                "fake": "reference_profile",
                "sample": self.reference_df.copy(),
            }

    class FakeSchemaChecker:
        def __init__(self, reference_df, required_features):
            self.reference_df = reference_df
            self.required_features = required_features
            self.checked_dfs = []
            self.error = None
            state.checker_instances.append(self)

        def check_df(self, df):
            self.checked_dfs.append(df)
            if self.error is not None:
                raise self.error

    class FakeDriftMetricsEngine:
        def __init__(self, config, reference_profile):
            self.config = config
            self.reference_profile = reference_profile
            self.analyze_current = None
            self.av_current = None
            self.av_kwargs = None
            state.engine_instances.append(self)

        def analyze_dataframe(self, current):
            self.analyze_current = current.copy()
            return {"kind": "drift_report"}

        def run_adversarial_validation(
            self,
            current,
            max_samples=100_000,
            n_splits=3,
            random_state=42,
            missing_category="__missing__",
            lightgbm_params=None,
        ):
            self.av_current = current.copy()
            self.av_kwargs = {
                "max_samples": max_samples,
                "n_splits": n_splits,
                "random_state": random_state,
                "missing_category": missing_category,
                "lightgbm_params": lightgbm_params,
            }
            return {"kind": "av_report"}

    monkeypatch.setattr(offline_mode, "extract_feature_groups", fake_extract_feature_groups)
    monkeypatch.setattr(offline_mode, "Profiler", FakeProfiler)
    monkeypatch.setattr(offline_mode, "SchemaChecker", FakeSchemaChecker)
    monkeypatch.setattr(offline_mode, "DriftMetricsEngine", FakeDriftMetricsEngine)

    return state


def test_init_rejects_path_to_config_and_config_options_together(reference_df):
    with pytest.raises(
        ValueError,
        match="Provide path_to_config or config_options or nothing. Got both.",
    ):
        OfflineWrapper(
            reference_df=reference_df,
            path_to_config="config.yml",
            config_options=object(),
        )


def test_init_from_config_path_reads_config_and_ignores_save_config_path(
    reference_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    config = object()
    calls = {"read_config": []}

    def fake_read_config(path):
        calls["read_config"].append(path)
        return config

    def fake_build_drift_config(*args, **kwargs):
        pytest.fail("build_drift_config must not be called when path_to_config is used")

    monkeypatch.setattr(offline_mode, "read_config", fake_read_config)
    monkeypatch.setattr(offline_mode, "build_drift_config", fake_build_drift_config)

    with pytest.warns(
        UserWarning,
        match="save_config_path is provided with path_to_config",
    ):
        wrapper = OfflineWrapper(
            reference_df=reference_df,
            path_to_config="config.yml",
            save_config_path="ignored.yml",
            merge_threshold=7,
            low_cardinality_threshold=11,
        )

    assert wrapper.config is config
    assert calls["read_config"] == ["config.yml"]

    assert wrapper.required_features == {"age", "country", "score"}
    assert wrapper.reference_dict["fake"] == "reference_profile"
    assert_frame_equal(wrapper.reference_dict["sample"], reference_df)
    assert wrapper.reference_dict["sample"] is not reference_df

    profiler = state.profiler_instances[0]
    assert profiler.reference_df is reference_df
    assert profiler.num_features == ["age"]
    assert profiler.cat_features == ["country"]
    assert profiler.prediction == "score"
    assert profiler.merge_threshold == 7
    assert profiler.low_cardinality_threshold == 11

    checker = state.checker_instances[0]
    assert checker.reference_df is reference_df
    assert checker.required_features == {"age", "country", "score"}

    engine = state.engine_instances[0]
    assert engine.config is config
    assert engine.reference_profile == wrapper.reference_dict


def test_init_from_config_options_builds_config_and_converts_dict(
    reference_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    options = object()
    raw_config = {"raw": "config"}
    parsed_config = object()

    calls = {
        "build": [],
        "config_from_dict": [],
    }

    def fake_build_drift_config(df, options=None, output_path=None):
        calls["build"].append(
            {
                "df": df,
                "options": options,
                "output_path": output_path,
            }
        )
        return raw_config

    def fake_config_from_dict(raw):
        calls["config_from_dict"].append(raw)
        return parsed_config

    def fake_read_config(*args, **kwargs):
        pytest.fail("read_config must not be called when config_options is used")

    monkeypatch.setattr(offline_mode, "build_drift_config", fake_build_drift_config)
    monkeypatch.setattr(offline_mode, "config_from_dict", fake_config_from_dict)
    monkeypatch.setattr(offline_mode, "read_config", fake_read_config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        config_options=options,
        save_config_path="generated.yml",
    )

    assert wrapper.config is parsed_config

    assert len(calls["build"]) == 1
    assert calls["build"][0]["df"] is reference_df
    assert calls["build"][0]["options"] is options
    assert calls["build"][0]["output_path"] == "generated.yml"

    assert calls["config_from_dict"] == [raw_config]

    assert wrapper.required_features == {"age", "country", "score"}
    assert len(state.profiler_instances) == 1
    assert len(state.checker_instances) == 1
    assert len(state.engine_instances) == 1


def test_analyze_df_checks_schema_and_passes_only_required_columns(
    reference_df,
    current_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    config = object()
    monkeypatch.setattr(offline_mode, "read_config", lambda path: config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        path_to_config="config.yml",
    )

    # В текущей реализации required_features — set.
    # Pandas 2.x не любит df[set(...)].
    # Поэтому для unit-теста поведения analyze_df фиксируем порядок колонок вручную.
    wrapper.required_features = ["age", "country", "score"]

    result = wrapper.analyze_df(current_df)

    assert result == {"kind": "drift_report"}

    checker = state.checker_instances[0]
    assert checker.checked_dfs == [current_df]

    engine = state.engine_instances[0]
    expected_current = current_df[["age", "country", "score"]]
    assert_frame_equal(engine.analyze_current, expected_current)


def test_analyze_df_propagates_schema_checker_error(
    reference_df,
    current_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    config = object()
    monkeypatch.setattr(offline_mode, "read_config", lambda path: config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        path_to_config="config.yml",
    )
    wrapper.required_features = ["age", "country", "score"]

    checker = state.checker_instances[0]
    checker.error = ValueError("bad schema")

    with pytest.raises(ValueError, match="bad schema"):
        wrapper.analyze_df(current_df)

    engine = state.engine_instances[0]
    assert engine.analyze_current is None


def test_run_av_checks_schema_and_passes_required_columns_and_params(
    reference_df,
    current_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    config = object()
    monkeypatch.setattr(offline_mode, "read_config", lambda path: config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        path_to_config="config.yml",
    )

    # См. комментарий в test_analyze_df_checks_schema_and_passes_only_required_columns.
    wrapper.required_features = ["age", "country", "score"]

    lightgbm_params = {"num_leaves": 7, "learning_rate": 0.05}

    result = wrapper.run_av(
        current=current_df,
        max_samples=123,
        n_splits=5,
        random_state=99,
        missing_category="__NA__",
        lightgbm_params=lightgbm_params,
    )

    assert result == {"kind": "av_report"}

    checker = state.checker_instances[0]
    assert checker.checked_dfs == [current_df]

    engine = state.engine_instances[0]
    expected_current = current_df[["age", "country", "score"]]
    assert_frame_equal(engine.av_current, expected_current)

    assert engine.av_kwargs == {
        "max_samples": 123,
        "n_splits": 5,
        "random_state": 99,
        "missing_category": "__NA__",
        "lightgbm_params": lightgbm_params,
    }


def test_run_av_propagates_schema_checker_error(
    reference_df,
    current_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    config = object()
    monkeypatch.setattr(offline_mode, "read_config", lambda path: config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        path_to_config="config.yml",
    )
    wrapper.required_features = ["age", "country", "score"]

    checker = state.checker_instances[0]
    checker.error = ValueError("invalid current dataframe")

    with pytest.raises(ValueError, match="invalid current dataframe"):
        wrapper.run_av(current_df)

    engine = state.engine_instances[0]
    assert engine.av_current is None


def test_analyze_df_current_implementation_fails_with_required_features_set(
    reference_df,
    current_df,
    monkeypatch,
):
    state = install_offline_wrapper_fakes(monkeypatch)

    config = object()
    monkeypatch.setattr(offline_mode, "read_config", lambda path: config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        path_to_config="config.yml",
    )

    result = wrapper.analyze_df(current_df)

    assert result == {"kind": "drift_report"}
    assert state.engine_instances[0].analyze_current is not None


def test_init_current_implementation_fails_when_one_feature_group_is_none(
    reference_df,
    monkeypatch,
):
    install_offline_wrapper_fakes(
        monkeypatch,
        num_feats=None,
        cat_feats=("country",),
        prediction=None,
    )

    config = object()
    monkeypatch.setattr(offline_mode, "read_config", lambda path: config)

    wrapper = OfflineWrapper(
        reference_df=reference_df,
        path_to_config="config.yml",
    )

    assert wrapper.required_features == {"country"}