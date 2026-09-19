import numpy as np
import pandas as pd
import pytest

from src.drift_guardian.analyzer.methods.low_level.stats import (
    psi,
    js_divergence,
    ks_d_statistic,
    chi2_p_value,
    cramer_v_measure_of_association,
    unseen_category_rate,
    category_churn,
    cardinality_ratio_abs_diff,
)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

class TestPSI:

    def test_identical_distributions_psi_close_to_zero(self):
        ref = [100, 200, 300, 400]
        actual = [100, 200, 300, 400]
        result = psi(ref, actual)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_psi_positive(self):
        ref = [100, 200, 300, 400]
        actual = [400, 300, 200, 100]
        result = psi(ref, actual)
        assert result > 0

    def test_psi_symmetric_property(self):
        # PSI не обязана быть симметричной по определению,
        # но проверим, что порядок аргументов даёт разный (но валидный) результат
        ref = [100, 200, 300, 400]
        actual = [150, 250, 250, 350]
        forward = psi(ref, actual)
        backward = psi(actual, ref)
        assert forward >= 0
        assert backward >= 0

    def test_psi_bin_cap_clipping(self):
        # Экстремальный перекос должен клипироваться bin_cap
        ref = [1000, 1, 1, 1]
        actual = [1, 1000, 1, 1]
        result = psi(ref, actual, bin_cap=0.5)
        # Максимально возможный psi при 4 бинах = 4 * bin_cap
        assert result <= 4 * 0.5 + 1e-6

    def test_psi_with_zero_counts_in_bin(self):
        # Один из бинов имеет 0 наблюдений — epsilon должен спасти от log(0)
        ref = [100, 0, 200, 300]
        actual = [90, 10, 190, 310]
        result = psi(ref, actual)
        assert np.isfinite(result)

    def test_psi_epsilon_affects_result(self):
        ref = [100, 0, 200, 300]
        actual = [90, 10, 190, 310]
        result_small_eps = psi(ref, actual, epsilon=1e-6)
        result_large_eps = psi(ref, actual, epsilon=1e-2)
        assert result_small_eps != pytest.approx(result_large_eps)

    def test_psi_returns_float(self):
        result = psi([10, 20, 30], [15, 15, 30])
        assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# JS Divergence
# ---------------------------------------------------------------------------

class TestJSDivergence:

    def test_identical_distributions_js_zero(self):
        ref = [0.25, 0.25, 0.25, 0.25]
        actual = [0.25, 0.25, 0.25, 0.25]
        result = js_divergence(ref, actual)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_completely_different_distributions_js_max(self):
        ref = [1, 0, 0, 0]
        actual = [0, 0, 0, 1]
        result = js_divergence(ref, actual, base=2)
        # JS divergence в квадрате при base=2 максимум = 1
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_js_divergence_is_non_negative(self):
        ref = [10, 30, 60]
        actual = [50, 20, 30]
        result = js_divergence(ref, actual)
        assert result >= 0

    @pytest.mark.parametrize("base", [2, np.e, 10])
    def test_js_divergence_different_bases(self, base):
        ref = [10, 30, 60]
        actual = [50, 20, 30]
        result = js_divergence(ref, actual, base=base)
        assert result >= 0
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# KS Statistic
# ---------------------------------------------------------------------------

class TestKSStatistic:

    def test_identical_samples_ks_zero(self):
        data = [1, 2, 3, 4, 5]
        result = ks_d_statistic(data, data)
        assert result == pytest.approx(0.0)

    def test_completely_separated_samples_ks_one(self):
        ref = [0, 0, 0, 0]
        actual = [10, 10, 10, 10]
        result = ks_d_statistic(ref, actual)
        assert result == pytest.approx(1.0)

    def test_ks_statistic_is_between_0_and_1(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 100)
        actual = np.random.normal(0.5, 1, 100)
        result = ks_d_statistic(ref, actual)
        assert 0 <= result <= 1

    def test_ks_statistic_increases_with_shift(self):
        np.random.seed(0)
        ref = np.random.normal(0, 1, 1000)
        actual_small_shift = np.random.normal(0.1, 1, 1000)
        actual_large_shift = np.random.normal(5, 1, 1000)

        small = ks_d_statistic(ref, actual_small_shift)
        large = ks_d_statistic(ref, actual_large_shift)
        assert large > small


# ---------------------------------------------------------------------------
# Chi2 p-value
# ---------------------------------------------------------------------------

class TestChi2PValue:

    def test_identical_distributions_high_p_value(self):
        ref = [100, 200, 300]
        actual = [100, 200, 300]
        result = chi2_p_value(ref, actual)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_very_different_distributions_low_p_value(self):
        ref = [1000, 10, 10]
        actual = [10, 1000, 10]
        result = chi2_p_value(ref, actual)
        assert result < 0.05

    def test_p_value_in_valid_range(self):
        ref = [50, 30, 20]
        actual = [45, 35, 20]
        result = chi2_p_value(ref, actual)
        assert 0 <= result <= 1

    def test_p_value_symmetric(self):
        ref = [50, 30, 20]
        actual = [45, 35, 20]
        p1 = chi2_p_value(ref, actual)
        p2 = chi2_p_value(actual, ref)
        assert p1 == pytest.approx(p2)


# ---------------------------------------------------------------------------
# Cramer's V
# ---------------------------------------------------------------------------

class TestCramerV:

    def test_identical_distributions_cramer_v_zero(self):
        ref = [100, 200, 300]
        actual = [100, 200, 300]
        result = cramer_v_measure_of_association(ref, actual)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_cramer_v_positive(self):
        ref = [1000, 10, 10]
        actual = [10, 1000, 10]
        result = cramer_v_measure_of_association(ref, actual)
        assert result > 0

    def test_cramer_v_in_valid_range(self):
        ref = [50, 30, 20]
        actual = [45, 35, 20]
        result = cramer_v_measure_of_association(ref, actual)
        assert 0 <= result <= 1


# ---------------------------------------------------------------------------
# Unseen Category Rate
# ---------------------------------------------------------------------------

class TestUnseenCategoryRate:

    def test_no_unseen_categories(self):
        ref = {"a": 10, "b": 20}
        actual = {"a": 5, "b": 15}
        result = unseen_category_rate(ref, actual)
        assert result == pytest.approx(0.0)

    def test_all_categories_unseen(self):
        ref = {"a": 10, "b": 20}
        actual = {"c": 5, "d": 15}
        result = unseen_category_rate(ref, actual)
        assert result == pytest.approx(1.0)

    def test_partial_unseen_categories(self):
        ref = {"a": 10, "b": 20}
        actual = {"a": 10, "c": 10}
        result = unseen_category_rate(ref, actual)
        # c=10 из общих 20 => 0.5
        assert result == pytest.approx(0.5)

    def test_empty_actual_freq_returns_zero(self):
        ref = {"a": 10}
        actual = {}
        result = unseen_category_rate(ref, actual)
        assert result == pytest.approx(0.0)

    def test_empty_ref_freq_all_unseen(self):
        ref = {}
        actual = {"a": 10, "b": 20}
        result = unseen_category_rate(ref, actual)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Category Churn
# ---------------------------------------------------------------------------

class TestCategoryChurn:

    def test_no_churn_identical_categories(self):
        ref = {"a": 10, "b": 20}
        actual = {"a": 5, "b": 15}
        result = category_churn(ref, actual)
        assert result == pytest.approx(0.0)

    def test_complete_churn_no_overlap(self):
        ref = {"a": 10, "b": 20}
        actual = {"c": 5, "d": 15}
        result = category_churn(ref, actual)
        # все 4 категории уникальны => churn = 4/4 = 1.0
        assert result == pytest.approx(1.0)

    def test_partial_churn(self):
        ref = {"a": 1, "b": 1, "c": 1}
        actual = {"b": 1, "c": 1, "d": 1}
        result = category_churn(ref, actual)
        # new={"d"}, disappeared={"a"}, all_cats={"a","b","c","d"}
        # => (1+1)/4 = 0.5
        assert result == pytest.approx(0.5)

    def test_empty_dicts_returns_zero(self):
        result = category_churn({}, {})
        assert result == pytest.approx(0.0)

    def test_new_categories_only(self):
        ref = {"a": 1}
        actual = {"a": 1, "b": 1}
        result = category_churn(ref, actual)
        # new={"b"}, disappeared={}, all_cats={"a","b"} => 1/2
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Cardinality Ratio Abs Diff
# ---------------------------------------------------------------------------

class TestCardinalityRatioAbsDiff:

    def test_zero_diff_when_ratios_match(self):
        current = pd.Series([1, 2, 3, 4, 5])
        ref_ratio = current.nunique() / len(current)
        result = cardinality_ratio_abs_diff(ref_ratio, current)
        assert result == pytest.approx(0.0)

    def test_positive_diff_when_ratios_differ(self):
        current = pd.Series([1, 1, 1, 2, 2])  # cardinality_ratio = 2/5 = 0.4
        ref_ratio = 0.8
        result = cardinality_ratio_abs_diff(ref_ratio, current)
        assert result == pytest.approx(0.4)

    def test_handles_nan_values_via_dropna(self):
        current = pd.Series([1, 2, 3, np.nan, np.nan])
        # после dropna: [1,2,3], nunique=3, len=3 => ratio=1.0
        ref_ratio = 1.0
        result = cardinality_ratio_abs_diff(ref_ratio, current)
        assert result == pytest.approx(0.0)

    def test_all_unique_values(self):
        current = pd.Series(range(100))  # cardinality_ratio = 1.0
        ref_ratio = 0.5
        result = cardinality_ratio_abs_diff(ref_ratio, current)
        assert result == pytest.approx(0.5)

    def test_all_duplicate_values(self):
        current = pd.Series([1] * 50)  # cardinality_ratio = 1/50 = 0.02
        ref_ratio = 1.0
        result = cardinality_ratio_abs_diff(ref_ratio, current)
        assert result == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# Parametrized edge-case тесты (общие паттерны)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "func,ref,actual",
    [
        (psi, [10, 20, 30], [10, 20, 30]),
        (js_divergence, [10, 20, 30], [10, 20, 30]),
        (chi2_p_value, [10, 20, 30], [10, 20, 30]),
        (cramer_v_measure_of_association, [10, 20, 30], [10, 20, 30]),
    ],
)
def test_metrics_near_zero_on_identical_input(func, ref, actual):
    """Общий smoke-тест: метрики дрейфа должны показывать
    отсутствие дрейфа при идентичных распределениях."""
    result = func(ref, actual)
    if func is chi2_p_value:
        assert result == pytest.approx(1.0, abs=1e-6)
    else:
        assert result == pytest.approx(0.0, abs=1e-6)
