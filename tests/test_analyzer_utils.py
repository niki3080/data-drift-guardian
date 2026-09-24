from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drift_guardian.config_handler.parse_config import FeatureType

from drift_guardian.analyzer.utils import (
    make_counts,
    find_ref,
    extract_feature_groups,
)


# ---------------------------------------------------------------------
# make_counts
# ---------------------------------------------------------------------


def test_make_counts_categorical_with_merged_rare_and_unseen_categories():
    reference = {
        "type": "categorical",
        "low_cardinality": False,
        "categories": {
            "a": 10,
            "b": 5,
            "rare": 1,
        },
        "merge_info": {
            "other_bucket": {
                "categories": ["rare"],
                "merge_cats_sum": 1,
            },
        },
    }
    current = pd.Series(["a", "a", "b", "rare", "new", None])

    ref_counts, cur_counts = make_counts(reference, current)

    assert ref_counts == [10, 5, 1]
    assert cur_counts == [2, 1, 2]


def test_make_counts_categorical_without_other_bucket_and_without_unseen_categories():
    reference = {
        "type": "categorical",
        "low_cardinality": False,
        "categories": {
            "a": 10,
            "b": 5,
        },
        "merge_info": {
            "other_bucket": {
                "categories": [],
                "merge_cats_sum": 0,
            },
        },
    }
    current = pd.Series(["a", "a", "b", None])

    ref_counts, cur_counts = make_counts(reference, current)

    assert ref_counts == [10, 5]
    assert cur_counts == [2, 1]


def test_make_counts_categorical_adds_zero_for_absent_current_categories():
    reference = {
        "type": "categorical",
        "low_cardinality": False,
        "categories": {
            "a": 10,
            "b": 5,
            "rare": 2,
        },
        "merge_info": {
            "other_bucket": {
                "categories": ["rare"],
                "merge_cats_sum": 2,
            },
        },
    }
    current = pd.Series(["a", "a"])

    ref_counts, cur_counts = make_counts(reference, current)

    assert ref_counts == [10, 5, 2]
    assert cur_counts == [2, 0, 0]


def test_make_counts_categorical_ignores_missing_values():
    reference = {
        "type": "categorical",
        "low_cardinality": False,
        "categories": {
            "a": 10,
            "b": 5,
        },
        "merge_info": {
            "other_bucket": {
                "categories": [],
                "merge_cats_sum": 0,
            },
        },
    }
    current = pd.Series(["a", None, np.nan, "b", "b"])

    ref_counts, cur_counts = make_counts(reference, current)

    assert ref_counts == [10, 5]
    assert cur_counts == [1, 2]


def test_make_counts_low_cardinality_numeric_without_other_bucket():
    reference = {
        "type": "numeric",
        "low_cardinality": True,
        "categories": {
            1: 10,
            2: 5,
            3: 2,
        },
        "merge_info": {
            "other_bucket": {
                "categories": [],
                "merge_cats_sum": 0,
            },
        },
    }
    current = pd.Series([1, 1, 2, 3, 3, 3, None])

    ref_counts, cur_counts = make_counts(reference, current)

    assert ref_counts == [10, 5, 2]
    assert cur_counts == [2, 1, 3]


def test_make_counts_numeric_uses_reference_decile_bins():
    reference = {
        "type": "numeric",
        "low_cardinality": False,
        "decile_bins": {
            "frequencies": np.array([2, 3, 5]),
            "deciles": [-np.inf, 0.0, 5.0, np.inf],
        },
    }
    current = pd.Series([-10.0, 1.0, 2.0, 6.0, np.nan])

    ref_counts, cur_counts = make_counts(reference, current)

    np.testing.assert_array_equal(ref_counts, np.array([2, 3, 5]))
    np.testing.assert_array_equal(cur_counts, np.array([1, 2, 1]))


def test_make_counts_numeric_ignores_missing_values():
    reference = {
        "type": "numeric",
        "low_cardinality": False,
        "decile_bins": {
            "frequencies": np.array([1, 1]),
            "deciles": [-np.inf, 0.0, np.inf],
        },
    }
    current = pd.Series([-1.0, np.nan, None, 1.0])

    ref_counts, cur_counts = make_counts(reference, current)

    np.testing.assert_array_equal(ref_counts, np.array([1, 1]))
    np.testing.assert_array_equal(cur_counts, np.array([1, 1]))


def test_make_counts_raises_for_unknown_feature_type():
    reference = {
        "type": "unknown",
        "low_cardinality": False,
    }
    current = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="Unknown feature type in reference: unknown"):
        make_counts(reference, current)


def test_make_counts_low_cardinality_numeric_with_other_bucket_currently_fails():
    reference = {
        "type": "numeric",
        "low_cardinality": True,
        "categories": {
            1: 10,
            2: 5,
            3: 1,
        },
        "merge_info": {
            "other_bucket": {
                "categories": [3],
                "merge_cats_sum": 1,
            },
        },
    }
    current = pd.Series([1, 2, 3, 999])

    ref_counts, cur_counts = make_counts(reference, current)

    assert ref_counts == [10, 5, 1]
    assert cur_counts == [1, 1, 2]


# ---------------------------------------------------------------------
# find_ref
# ---------------------------------------------------------------------


def test_find_ref_returns_numeric_reference():
    numeric_reference = {
        "type": "numeric",
        "feature": "age",
    }
    reference_dict = {
        "num_ref": {
            "age": numeric_reference,
        },
        "cat_ref": {},
        "preds_ref": {},
    }

    result = find_ref(reference_dict, "age")

    assert result is numeric_reference


def test_find_ref_returns_categorical_reference():
    categorical_reference = {
        "type": "categorical",
        "feature": "city",
    }
    reference_dict = {
        "num_ref": {},
        "cat_ref": {
            "city": categorical_reference,
        },
        "preds_ref": {},
    }

    result = find_ref(reference_dict, "city")

    assert result is categorical_reference


def test_find_ref_returns_prediction_reference():
    prediction_reference = {
        "type": "numeric",
        "feature": "score",
    }
    reference_dict = {
        "num_ref": {},
        "cat_ref": {},
        "preds_ref": {
            "score": prediction_reference,
        },
    }

    result = find_ref(reference_dict, "score")

    assert result is prediction_reference


def test_find_ref_prefers_numeric_reference_over_categorical_and_prediction_reference():
    numeric_reference = {
        "type": "numeric",
        "source": "num_ref",
    }
    categorical_reference = {
        "type": "categorical",
        "source": "cat_ref",
    }
    prediction_reference = {
        "type": "numeric",
        "source": "preds_ref",
    }
    reference_dict = {
        "num_ref": {
            "same_feature": numeric_reference,
        },
        "cat_ref": {
            "same_feature": categorical_reference,
        },
        "preds_ref": {
            "same_feature": prediction_reference,
        },
    }

    result = find_ref(reference_dict, "same_feature")

    assert result is numeric_reference


def test_find_ref_prefers_categorical_reference_over_prediction_reference():
    categorical_reference = {
        "type": "categorical",
        "source": "cat_ref",
    }
    prediction_reference = {
        "type": "numeric",
        "source": "preds_ref",
    }
    reference_dict = {
        "num_ref": {},
        "cat_ref": {
            "same_feature": categorical_reference,
        },
        "preds_ref": {
            "same_feature": prediction_reference,
        },
    }

    result = find_ref(reference_dict, "same_feature")

    assert result is categorical_reference


def test_find_ref_raises_if_feature_not_found():
    reference_dict = {
        "num_ref": {},
        "cat_ref": {},
        "preds_ref": {},
    }

    with pytest.raises(ValueError, match="No feature: missing_feature in reference_dict"):
        find_ref(reference_dict, "missing_feature")


# ---------------------------------------------------------------------
# extract_feature_groups
# ---------------------------------------------------------------------


def test_extract_feature_groups_returns_numeric_categorical_and_prediction():
    config = SimpleNamespace(
        features={
            "age": SimpleNamespace(type=FeatureType.numeric),
            "income": SimpleNamespace(type=FeatureType.numeric),
            "city": SimpleNamespace(type=FeatureType.categorical),
            "segment": SimpleNamespace(type=FeatureType.categorical),
        },
        prediction_metrics=SimpleNamespace(
            enabled=True,
            score_column="score",
        ),
    )

    num_features, cat_features, prediction = extract_feature_groups(config)

    assert num_features == ["age", "income"]
    assert cat_features == ["city", "segment"]
    assert prediction == "score"


def test_extract_feature_groups_returns_none_for_empty_feature_groups():
    config = SimpleNamespace(
        features={},
        prediction_metrics=SimpleNamespace(
            enabled=False,
            score_column="score",
        ),
    )

    num_features, cat_features, prediction = extract_feature_groups(config)

    assert num_features is None
    assert cat_features is None
    assert prediction is None


def test_extract_feature_groups_returns_only_numeric_features():
    config = SimpleNamespace(
        features={
            "age": SimpleNamespace(type=FeatureType.numeric),
            "income": SimpleNamespace(type=FeatureType.numeric),
        },
        prediction_metrics=SimpleNamespace(
            enabled=False,
            score_column="score",
        ),
    )

    num_features, cat_features, prediction = extract_feature_groups(config)

    assert num_features == ["age", "income"]
    assert cat_features is None
    assert prediction is None


def test_extract_feature_groups_returns_only_categorical_features():
    config = SimpleNamespace(
        features={
            "city": SimpleNamespace(type=FeatureType.categorical),
            "segment": SimpleNamespace(type=FeatureType.categorical),
        },
        prediction_metrics=SimpleNamespace(
            enabled=False,
            score_column="score",
        ),
    )

    num_features, cat_features, prediction = extract_feature_groups(config)

    assert num_features is None
    assert cat_features == ["city", "segment"]
    assert prediction is None


def test_extract_feature_groups_returns_prediction_when_enabled_even_without_features():
    config = SimpleNamespace(
        features={},
        prediction_metrics=SimpleNamespace(
            enabled=True,
            score_column="prediction_score",
        ),
    )

    num_features, cat_features, prediction = extract_feature_groups(config)

    assert num_features is None
    assert cat_features is None
    assert prediction == "prediction_score"


def test_extract_feature_groups_ignores_prediction_score_when_prediction_metrics_disabled():
    config = SimpleNamespace(
        features={
            "age": SimpleNamespace(type=FeatureType.numeric),
        },
        prediction_metrics=SimpleNamespace(
            enabled=False,
            score_column="score",
        ),
    )

    num_features, cat_features, prediction = extract_feature_groups(config)

    assert num_features == ["age"]
    assert cat_features is None
    assert prediction is None
