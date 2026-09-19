import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.drift_guardian.profiler.baseline_profiler import Profiler


WINDOW_SIZE_NO_SAMPLE_WARNING = "If take_sample=False, window_size does nothing"


@pytest.fixture
def ref_df():
    return pd.DataFrame(
        {
            "num": [1.0, 2.5, 3.0, 4.5, 5.0, 6.0],
            "num_int": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "a", "b", "c", "d", None],
            "bool_col": [True, False, True, False, True, False],
            "str_col": ["1", "2", "3", "4", "5", "6"],
            "dt": pd.date_range("2024-01-01", periods=6),
            "target": [0.0, 1.0, 0.0, 1.0, np.nan, 1.0],
        }
    )


def test_init_warns_if_window_size_set_when_take_sample_is_false(ref_df):
    with pytest.warns(UserWarning, match=re.escape(WINDOW_SIZE_NO_SAMPLE_WARNING)):
        profiler = Profiler(
            ref_data=ref_df,
            window_size=3,
            num_features=["num"],
            take_sample=False,
        )

    assert profiler.num_features == ["num"]


def test_init_success_with_explicit_features(ref_df):
    profiler = Profiler(
        ref_data=ref_df,
        window_size=None,
        num_features=["num"],
        cat_features=["cat"],
        take_sample=False,
    )

    assert profiler.window_size is None
    assert profiler.num_features == ["num"]
    assert profiler.cat_features == ["cat"]
    assert profiler.prediction is None
    assert list(profiler.ref_data.columns) == ["num", "cat"]


def test_init_adds_numeric_prediction_to_num_features(ref_df):
    profiler = Profiler(
        ref_data=ref_df,
        window_size=None,
        num_features=["num"],
        cat_features=["cat"],
        prediction="target",
        take_sample=False,
    )

    assert "target" in profiler.num_features
    assert "target" not in profiler.cat_features
    assert list(profiler.ref_data.columns) == ["num", "target", "cat"]


def test_init_adds_categorical_prediction_to_cat_features(ref_df):
    df = ref_df.assign(pred_label=["x", "y", "x", "y", "x", None])

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["num"],
        cat_features=["cat"],
        prediction="pred_label",
        take_sample=False,
    )

    assert "pred_label" in profiler.cat_features
    assert "pred_label" not in profiler.num_features
    assert list(profiler.ref_data.columns) == ["num", "cat", "pred_label"]


def test_init_ref_data_must_be_dataframe():
    with pytest.raises(ValueError, match="ref_data must be provided in pandas DataFrame format"):
        Profiler(
            ref_data={"num": [1, 2, 3]},
            window_size=3,
            num_features=["num"],
            take_sample=True,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_features": [1]}, "All num_features must be strings"),
        ({"cat_features": [1]}, "All cat_features must be strings"),
        ({"num_features": ["num", "num"]}, "Duplicate feature"),
        ({"cat_features": ["cat", "cat"]}, "Duplicate feature"),
        (
            {"num_features": ["num"], "cat_features": ["num"]},
            "provided in both num_features and cat_features",
        ),
        ({"prediction": 123}, "prediction must be string or None"),
        ({"window_size": 0}, "window_size must be greater than 0"),
        ({"window_size": True}, "window_size must be int"),
        ({"window_size": 1.5}, "window_size must be int"),
        ({"merge_threshold": 0}, "merge_threshold must be greater than 0"),
        ({"merge_threshold": True}, "merge_threshold must be int"),
        (
            {"low_cardinality_threshold": 0},
            "low_cardinality_threshold must be greater than 0",
        ),
        (
            {"low_cardinality_threshold": True},
            "low_cardinality_threshold must be int",
        ),
        ({"take_sample": 1}, "sample must be bool"),
        ({"sample_float_dtype": 123}, "sample_dtype must be str"),
        ({"sample_float_dtype": "not-a-dtype"}, "Invalid numpy dtype"),
        ({"sample_float_dtype": "int64"}, "sample_dtype must be a numpy floating dtype"),
        ({"random_state": True}, "random_state must be int or None"),
        ({"random_state": "42"}, "random_state must be int or None"),
    ],
)
def test_init_validation_errors(ref_df, overrides, match):
    kwargs = {
        "ref_data": ref_df,
        "window_size": 3,
        "num_features": ["num"],
        "cat_features": ["cat"],
        "take_sample": True,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=re.escape(match)):
        Profiler(**kwargs)


def test_init_requires_at_least_one_feature_or_prediction(ref_df):
    with pytest.raises(ValueError, match="Nothing to profile"):
        Profiler(
            ref_data=ref_df,
            window_size=3,
            take_sample=True,
        )


def test_init_reports_missing_columns(ref_df):
    with pytest.raises(ValueError) as exc_info:
        Profiler(
            ref_data=ref_df,
            window_size=3,
            num_features=["missing_num"],
            cat_features=["missing_cat"],
            prediction="missing_pred",
            take_sample=True,
        )

    message = str(exc_info.value)

    assert "Prediction: ['missing_pred'] is missing in ref_data" in message
    assert "num_feature(s): ['missing_num'] is missing in ref_data" in message
    assert "cat_feature(s): ['missing_cat'] is missing in ref_data" in message


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"num_features": ["dt"], "cat_features": []},
            "Num feature: dt has dtype",
        ),
        (
            {"num_features": [], "cat_features": ["dt"]},
            "Cat feature: dt has dtype",
        ),
        (
            {"num_features": [], "cat_features": [], "prediction": "dt"},
            "Prediction col: dt has dtype",
        ),
    ],
)
def test_time_dtypes_are_rejected(ref_df, kwargs, match):
    base_kwargs = {
        "ref_data": ref_df,
        "window_size": 3,
        "take_sample": True,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=re.escape(match)):
        Profiler(**base_kwargs)


@pytest.mark.parametrize(
    ("num_col", "match"),
    [
        ("cat", "doesn't have numeric dtype"),
        ("bool_col", "has bool dtype"),
    ],
)
def test_num_features_must_be_numeric_and_not_bool(ref_df, num_col, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        Profiler(
            ref_data=ref_df,
            window_size=3,
            num_features=[num_col],
            take_sample=True,
        )


def test_profile_cat_feature(ref_df):
    profiler = Profiler(
        ref_data=ref_df,
        window_size=None,
        cat_features=["cat"],
        merge_threshold=2,
        take_sample=False,
    )

    result = profiler.profile_ref_data()
    cat_profile = result["cat_ref"]["cat"]

    assert cat_profile["feature"] == "cat"
    assert cat_profile["type"] == "categorical"
    assert cat_profile["n"] == 5
    assert cat_profile["missing_rate"] == pytest.approx(1 / 6)
    assert cat_profile["cardinality_ratio"] == pytest.approx(4 / 5)

    assert cat_profile["categories"] == {
        "a": 2,
        "b": 1,
        "c": 1,
        "d": 1,
    }

    assert cat_profile["proportions"] == {
        "a": pytest.approx(2 / 5),
        "b": pytest.approx(1 / 5),
        "c": pytest.approx(1 / 5),
        "d": pytest.approx(1 / 5),
    }

    assert cat_profile["is_complete_category_list"] is True
    assert cat_profile["churn_baseline"] == "reference"

    merge_info = cat_profile["merge_info"]
    assert merge_info["merge_threshold"] == 2
    assert set(merge_info["other_bucket"]["categories"]) == {"b", "c", "d"}
    assert merge_info["other_bucket"]["is_catch_all_for_unseen"] is True
    assert merge_info["other_bucket"]["merge_cats_sum"] == 3


def test_profile_high_cardinality_numeric_feature():
    df = pd.DataFrame({"x": np.arange(1, 101, dtype=float)})

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["x"],
        low_cardinality_threshold=10,
        take_sample=False,
    )

    result = profiler.profile_ref_data()
    num_profile = result["num_ref"]["x"]

    assert num_profile["type"] == "numeric"
    assert num_profile["n"] == 100
    assert num_profile["missing_rate"] == 0
    assert num_profile["mean"] == pytest.approx(50.5)
    assert num_profile["min"] == 1.0
    assert num_profile["max"] == 100.0
    assert num_profile["low_cardinality"] is False

    assert set(num_profile["quantiles"]) == {
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }

    decile_bins = num_profile["decile_bins"]
    assert sum(decile_bins["frequencies"]) == 100
    assert np.isneginf(decile_bins["deciles"][0])
    assert np.isposinf(decile_bins["deciles"][-1])


def test_profile_low_cardinality_numeric_feature():
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3, np.nan]})

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["x"],
        low_cardinality_threshold=10,
        merge_threshold=2,
        take_sample=False,
    )

    result = profiler.profile_ref_data()
    num_profile = result["num_ref"]["x"]

    assert num_profile["type"] == "numeric"
    assert num_profile["n"] == 5
    assert num_profile["missing_rate"] == pytest.approx(1 / 6)
    assert num_profile["low_cardinality"] is True

    assert "decile_bins" not in num_profile
    assert num_profile["categories"] == {
        1.0: 2,
        2.0: 2,
        3.0: 1,
    }
    assert num_profile["proportions"] == {
        1.0: pytest.approx(2 / 5),
        2.0: pytest.approx(2 / 5),
        3.0: pytest.approx(1 / 5),
    }

    assert set(num_profile["merge_info"]["other_bucket"]["categories"]) == {3.0}
    assert num_profile["merge_info"]["other_bucket"]["merge_cats_sum"] == 1


def test_prediction_profile_is_separated_from_feature_refs(ref_df):
    profiler = Profiler(
        ref_data=ref_df,
        window_size=None,
        num_features=["num"],
        cat_features=["cat"],
        prediction="target",
        low_cardinality_threshold=10,
        take_sample=False,
    )

    with pytest.warns(UserWarning, match="Missing values in prediction"):
        result = profiler.profile_ref_data()

    assert "target" not in result["num_ref"]
    assert "target" not in result["cat_ref"]

    assert result["preds_ref"]["type"] == "num"
    assert "target" in result["preds_ref"]
    assert result["preds_ref"]["target"]["missing_rate"] == pytest.approx(1 / 6)


def test_categorical_prediction_profile_is_separated(ref_df):
    df = ref_df.assign(pred_label=["x", "y", "x", "y", "x", "y"])

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["num"],
        cat_features=["cat"],
        prediction="pred_label",
        take_sample=False,
    )

    result = profiler.profile_ref_data()

    assert result["preds_ref"]["type"] == "cat"
    assert "pred_label" in result["preds_ref"]
    assert "pred_label" not in result["cat_ref"]


def test_profile_empty_numeric_column_raises_value_error():
    df = pd.DataFrame({"x": pd.Series(dtype="float64")})

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["x"],
        take_sample=False,
    )

    with pytest.raises(ValueError, match="Column 'x' is empty"):
        profiler.profile_ref_data()


def test_profile_all_missing_numeric_column_raises_value_error():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["x"],
        take_sample=False,
    )

    with pytest.raises(ValueError, match="All values in column 'x' is missing"):
        profiler.profile_ref_data()


def test_build_reference_sample_returns_deterministic_sample():
    df = pd.DataFrame({"x": np.arange(100)})

    sample_1 = Profiler.build_reference_sample(
        full_values=df,
        window_size=4,
        min_size=5,
        multiplier=2,
        random_state=42,
    )
    sample_2 = Profiler.build_reference_sample(
        full_values=df,
        window_size=4,
        min_size=5,
        multiplier=2,
        random_state=42,
    )

    assert len(sample_1) == 8
    assert_frame_equal(sample_1, sample_2)


def test_build_reference_sample_returns_full_df_if_it_is_small():
    df = pd.DataFrame({"x": np.arange(5)})

    sample = Profiler.build_reference_sample(
        full_values=df,
        window_size=10,
        min_size=5,
        multiplier=2,
        random_state=42,
    )

    assert_frame_equal(sample, df)


def test_compress_df_downcasts_numeric_and_converts_cat_to_category():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a"],
            "pos_int": [1, 2, 3],
            "neg_int": [-1, 0, 1],
            "float_col": [1.1, 2.2, 3.3],
        }
    )

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        num_features=["pos_int", "neg_int", "float_col"],
        cat_features=["cat"],
        sample_float_dtype="float32",
        take_sample=False,
    )

    compressed = profiler.compress_df(df)

    assert str(compressed["cat"].dtype) == "category"

    assert np.issubdtype(compressed["pos_int"].dtype, np.unsignedinteger)
    assert np.issubdtype(compressed["neg_int"].dtype, np.signedinteger)
    assert compressed["float_col"].dtype == np.dtype("float32")


def test_profile_ref_data_result_has_expected_top_level_keys(ref_df):
    profiler = Profiler(
        ref_data=ref_df,
        window_size=None,
        num_features=["num"],
        cat_features=["cat"],
        prediction="target",
        take_sample=False,
    )

    with pytest.warns(UserWarning):
        result = profiler.profile_ref_data()

    assert set(result.keys()) == {
        "cat_ref",
        "num_ref",
        "sample",
        "preds_ref",
    }

    assert isinstance(result["cat_ref"], dict)
    assert isinstance(result["num_ref"], dict)
    assert isinstance(result["sample"], pd.DataFrame)
    assert isinstance(result["preds_ref"], dict)


def test_profile_all_missing_categorical_column_raises_value_error():
    df = pd.DataFrame({"cat": [None, None, None]})

    profiler = Profiler(
        ref_data=df,
        window_size=None,
        cat_features=["cat"],
        take_sample=False,
    )

    with pytest.raises(ValueError, match="All values in column 'cat' is missing"):
        profiler.profile_ref_data()
