import numpy as np
import pandas as pd
import pytest

from src.drift_guardian.analyzer.methods import cardinality_ratio
from src.drift_guardian.analyzer.methods import js_divergence
from src.drift_guardian.analyzer.methods import chi2
from src.drift_guardian.analyzer.methods import psi
from src.drift_guardian.analyzer.methods import missing_rate
from src.drift_guardian.analyzer.methods import cramer_v
from src.drift_guardian.analyzer.methods import ks_d_statistic
from src.drift_guardian.analyzer.methods import wn_distance
from src.drift_guardian.analyzer.methods import unseen_category_rate
from src.drift_guardian.analyzer.methods import category_churn


# ---------------------------------------------------------------------
# compute_cardinality_ratio_abs_diff
# ---------------------------------------------------------------------


def test_compute_cardinality_ratio_abs_diff_calls_find_ref_and_low_level(monkeypatch):
    reference_dict = {"some": "reference_dict"}
    reference = {"cardinality_ratio": 0.25}
    current = pd.Series(["a", "b", "c"], name="cat_feature")

    calls = {}

    def fake_find_ref(ref_dict, feature):
        calls["find_ref"] = {
            "reference_dict": ref_dict,
            "feature": feature,
        }
        return reference

    def fake_cardinality_ratio_abs_diff(ref_cardinality_ratio, cur):
        calls["low_level"] = {
            "ref_cardinality_ratio": ref_cardinality_ratio,
            "current": cur,
        }
        return 0.123

    monkeypatch.setattr(cardinality_ratio, "find_ref", fake_find_ref)
    monkeypatch.setattr(
        cardinality_ratio,
        "cardinality_ratio_abs_diff",
        fake_cardinality_ratio_abs_diff,
    )

    result = cardinality_ratio.compute_cardinality_ratio_abs_diff(
        reference_dict,
        current,
    )

    assert result == 0.123

    assert calls["find_ref"]["reference_dict"] is reference_dict
    assert calls["find_ref"]["feature"] == "cat_feature"

    assert calls["low_level"]["ref_cardinality_ratio"] == 0.25
    assert calls["low_level"]["current"] is current


def test_compute_cardinality_ratio_abs_diff_requires_named_series():
    with pytest.raises(AssertionError):
        cardinality_ratio.compute_cardinality_ratio_abs_diff(
            {},
            pd.Series(["a", "b"]),
        )


# ---------------------------------------------------------------------
# make_counts-based metrics:
# js_divergence, chi2, psi, cramer_v
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, func_name, low_level_name, expected_value",
    [
        (
            js_divergence,
            "compute_js_divergence",
            "js_divergence",
            0.11,
        ),
        (
            chi2,
            "compute_chi2_p_value",
            "chi2_p_value",
            0.22,
        ),
        (
            psi,
            "compute_psi",
            "psi",
            0.33,
        ),
        (
            cramer_v,
            "compute_cramer_v",
            "cramer_v_measure_of_association",
            0.44,
        ),
    ],
)
def test_make_counts_based_metric_calls_find_ref_make_counts_and_low_level(
    monkeypatch,
    module,
    func_name,
    low_level_name,
    expected_value,
):
    reference_dict = {"some": "reference_dict"}
    reference = {"type": "categorical", "categories": {"a": 10, "b": 5}}
    current = pd.Series(["a", "b", "b", "c"], name="feature_name")

    ref_counts = np.array([10, 5, 0])
    cur_counts = np.array([1, 2, 1])

    calls = {}

    def fake_find_ref(ref_dict, feature):
        calls["find_ref"] = {
            "reference_dict": ref_dict,
            "feature": feature,
        }
        return reference

    def fake_make_counts(ref, cur):
        calls["make_counts"] = {
            "reference": ref,
            "current": cur,
        }
        return ref_counts, cur_counts

    def fake_low_level(rc, cc):
        calls["low_level"] = {
            "ref_counts": rc,
            "cur_counts": cc,
        }
        return expected_value

    monkeypatch.setattr(module, "find_ref", fake_find_ref)
    monkeypatch.setattr(module, "make_counts", fake_make_counts)
    monkeypatch.setattr(module, low_level_name, fake_low_level)

    result = getattr(module, func_name)(reference_dict, current)

    assert result == expected_value

    assert calls["find_ref"]["reference_dict"] is reference_dict
    assert calls["find_ref"]["feature"] == "feature_name"

    assert calls["make_counts"]["reference"] is reference
    assert calls["make_counts"]["current"] is current

    assert calls["low_level"]["ref_counts"] is ref_counts
    assert calls["low_level"]["cur_counts"] is cur_counts


@pytest.mark.parametrize(
    "module, func_name",
    [
        (js_divergence, "compute_js_divergence"),
        (chi2, "compute_chi2_p_value"),
        (psi, "compute_psi"),
        (cramer_v, "compute_cramer_v"),
    ],
)
def test_make_counts_based_metric_requires_named_series(module, func_name):
    with pytest.raises(AssertionError):
        getattr(module, func_name)(
            {},
            pd.Series([1, 2, 3]),
        )


# ---------------------------------------------------------------------
# compute_missing_rate
# ---------------------------------------------------------------------


def test_compute_missing_rate_calls_find_ref(monkeypatch):
    reference_dict = {"some": "reference_dict"}
    reference = {"missing_rate": 0.37}
    current = pd.Series([1, None, 3, None], name="some_feature")

    calls = {}

    def fake_find_ref(ref_dict, feature):
        calls["find_ref"] = {
            "reference_dict": ref_dict,
            "feature": feature,
        }
        return reference

    monkeypatch.setattr(missing_rate, "find_ref", fake_find_ref)

    result = missing_rate.compute_missing_rate(reference_dict, current)

    assert result == 0.37

    assert calls["find_ref"]["reference_dict"] is reference_dict
    assert calls["find_ref"]["feature"] == "some_feature"


def test_compute_missing_rate_requires_named_series():
    with pytest.raises(AssertionError):
        missing_rate.compute_missing_rate(
            {},
            pd.Series([1, None, 3]),
        )


# ---------------------------------------------------------------------
# compute_unseen_category_rate
# ---------------------------------------------------------------------


def test_compute_unseen_category_rate_calls_find_ref_and_low_level(monkeypatch):
    reference_dict = {"some": "reference_dict"}
    reference = {
        "categories": {
            "a": 10,
            "b": 5,
        }
    }
    current = pd.Series(["a", "b", "b", "new", None], name="cat_feature")

    calls = {}

    def fake_find_ref(ref_dict, feature):
        calls["find_ref"] = {
            "reference_dict": ref_dict,
            "feature": feature,
        }
        return reference

    def fake_unseen_category_rate(reference_freq, current_freq):
        calls["low_level"] = {
            "reference_freq": reference_freq,
            "current_freq": current_freq,
        }
        return 0.25

    monkeypatch.setattr(unseen_category_rate, "find_ref", fake_find_ref)
    monkeypatch.setattr(
        unseen_category_rate,
        "unseen_category_rate",
        fake_unseen_category_rate,
    )

    result = unseen_category_rate.compute_unseen_category_rate(
        reference_dict,
        current,
    )

    assert result == 0.25

    assert calls["find_ref"]["reference_dict"] is reference_dict
    assert calls["find_ref"]["feature"] == "cat_feature"

    assert calls["low_level"]["reference_freq"] == {
        "a": 10,
        "b": 5,
    }
    assert calls["low_level"]["current_freq"] == {
        "b": 2,
        "a": 1,
        "new": 1,
    }


def test_compute_unseen_category_rate_requires_named_series():
    with pytest.raises(AssertionError):
        unseen_category_rate.compute_unseen_category_rate(
            {},
            pd.Series(["a", "b"]),
        )


# ---------------------------------------------------------------------
# compute_category_churn
# ---------------------------------------------------------------------


def test_compute_category_churn_calls_find_ref_and_low_level(monkeypatch):
    reference_dict = {"some": "reference_dict"}
    reference = {
        "categories": {
            "old": 10,
            "stable": 5,
        }
    }
    current = pd.Series(["stable", "stable", "new", None], name="cat_feature")

    calls = {}

    def fake_find_ref(ref_dict, feature):
        calls["find_ref"] = {
            "reference_dict": ref_dict,
            "feature": feature,
        }
        return reference

    def fake_category_churn(reference_freq, current_freq):
        calls["low_level"] = {
            "reference_freq": reference_freq,
            "current_freq": current_freq,
        }
        return 0.5

    monkeypatch.setattr(category_churn, "find_ref", fake_find_ref)
    monkeypatch.setattr(
        category_churn,
        "category_churn",
        fake_category_churn,
    )

    result = category_churn.compute_category_churn(
        reference_dict,
        current,
    )

    assert result == 0.5

    assert calls["find_ref"]["reference_dict"] is reference_dict
    assert calls["find_ref"]["feature"] == "cat_feature"

    assert calls["low_level"]["reference_freq"] == {
        "old": 10,
        "stable": 5,
    }
    assert calls["low_level"]["current_freq"] == {
        "stable": 2,
        "new": 1,
    }


def test_compute_category_churn_requires_named_series():
    with pytest.raises(AssertionError):
        category_churn.compute_category_churn(
            {},
            pd.Series(["a", "b"]),
        )


# ---------------------------------------------------------------------
# compute_ks_d_statistic
# ---------------------------------------------------------------------


def test_compute_ks_d_statistic_uses_raw_reference_sample_without_missing_values(
    monkeypatch,
):
    reference_dict = {
        "sample": pd.DataFrame(
            {
                "score": [1.0, np.nan, 3.0, 4.0],
            }
        )
    }
    current = pd.Series([2.0, 5.0, np.nan], name="score")

    calls = {}

    def fake_ks_d_statistic(raw_ref, cur):
        calls["low_level"] = {
            "raw_ref": raw_ref,
            "current": cur,
        }
        return 0.777

    monkeypatch.setattr(
        ks_d_statistic,
        "ks_d_statistic",
        fake_ks_d_statistic,
    )

    result = ks_d_statistic.compute_ks_d_statistic(
        reference_dict,
        current,
    )

    assert result == 0.777

    assert calls["low_level"]["raw_ref"].tolist() == [1.0, 3.0, 4.0]
    assert calls["low_level"]["current"] is current


def test_compute_ks_d_statistic_raises_key_error_if_feature_absent_in_sample():
    reference_dict = {
        "sample": pd.DataFrame(
            {
                "another_score": [1.0, 2.0, 3.0],
            }
        )
    }
    current = pd.Series([1.0, 2.0], name="score")

    with pytest.raises(KeyError):
        ks_d_statistic.compute_ks_d_statistic(
            reference_dict,
            current,
        )


def test_compute_ks_d_statistic_requires_named_series():
    with pytest.raises(AssertionError):
        ks_d_statistic.compute_ks_d_statistic(
            {"sample": pd.DataFrame({"score": [1.0, 2.0]})},
            pd.Series([1.0, 2.0]),
        )


# ---------------------------------------------------------------------
# compute_wasserstein_distance
# ---------------------------------------------------------------------


def test_compute_wasserstein_distance_uses_raw_reference_sample_without_missing_values(
    monkeypatch,
):
    reference_dict = {
        "sample": pd.DataFrame(
            {
                "amount": [10.0, np.nan, 30.0, 50.0],
            }
        )
    }
    current = pd.Series([20.0, 40.0], name="amount")

    calls = {}

    def fake_wasserstein_distance(raw_ref, cur):
        calls["wasserstein_distance"] = {
            "raw_ref": raw_ref,
            "current": cur,
        }
        return 12.34

    monkeypatch.setattr(
        wn_distance,
        "wasserstein_distance",
        fake_wasserstein_distance,
    )

    result = wn_distance.compute_wasserstein_distance(
        reference_dict,
        current,
    )

    assert result == 12.34

    assert calls["wasserstein_distance"]["raw_ref"].tolist() == [
        10.0,
        30.0,
        50.0,
    ]
    assert calls["wasserstein_distance"]["current"] is current


def test_compute_wasserstein_distance_raises_key_error_if_feature_absent_in_sample():
    reference_dict = {
        "sample": pd.DataFrame(
            {
                "another_amount": [1.0, 2.0, 3.0],
            }
        )
    }
    current = pd.Series([1.0, 2.0], name="amount")

    with pytest.raises(KeyError):
        wn_distance.compute_wasserstein_distance(
            reference_dict,
            current,
        )


def test_compute_wasserstein_distance_requires_named_series():
    with pytest.raises(AssertionError):
        wn_distance.compute_wasserstein_distance(
            {"sample": pd.DataFrame({"amount": [1.0, 2.0]})},
            pd.Series([1.0, 2.0]),
        )
