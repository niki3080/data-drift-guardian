import logging

import pandas as pd
from scipy.spatial import distance
from scipy.stats import chi2_contingency, kstest
from scipy.stats.contingency import association


import numpy as np

logger = logging.getLogger(__name__)


def psi(ref_counts, actual_counts, epsilon=1e-4, bin_cap=2.0):
    logger.debug(
        "psi: n_bins=%d, epsilon=%s, bin_cap=%s",
        len(ref_counts),
        epsilon,
        bin_cap,
    )
    n_bins = len(ref_counts)

    ref_total = sum(ref_counts)
    actual_total = sum(actual_counts)
    if ref_total == 0 or actual_total == 0:
        logger.warning(
            "psi: zero total counts detected (ref_total=%s, actual_total=%s), "
            "division by zero will occur",
            ref_total,
            actual_total,
        )

    expected_pct = np.array(ref_counts) / ref_total
    actual_pct = np.array(actual_counts) / actual_total

    expected_pct = (expected_pct + epsilon) / (1 + epsilon * n_bins)
    actual_pct = (actual_pct + epsilon) / (1 + epsilon * n_bins)

    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    clipped_count = np.sum(
        (psi_components > bin_cap) | (psi_components < -bin_cap)
    )
    if clipped_count:
        logger.debug(
            "psi: clipping %d/%d bin components to [-%s, %s]",
            clipped_count,
            n_bins,
            bin_cap,
            bin_cap,
        )
    psi_components = np.clip(psi_components, -bin_cap, bin_cap)

    result = np.sum(psi_components)
    logger.debug("psi: computed value=%.6f", result)

    return result


def js_divergence(ref_counts, actual_counts, base=2):
    logger.debug("js_divergence: base=%s, n_bins=%d", base, len(ref_counts))

    js_distance = distance.jensenshannon(ref_counts, actual_counts, base=base)
    result = js_distance ** 2

    logger.debug("js_divergence: computed value=%.6f", result)
    return result


def ks_d_statistic(raw_ref, raw_actual):
    logger.debug(
        "ks_d_statistic: ref_size=%d, actual_size=%d",
        len(raw_ref),
        len(raw_actual),
    )

    test_results = kstest(raw_ref, raw_actual)

    logger.debug(
        "ks_d_statistic: statistic=%.6f, p_value=%.6f",
        test_results.statistic,
        test_results.pvalue,
    )
    return test_results.statistic


def chi2_p_value(ref_counts, actual_counts):
    logger.debug("chi2_p_value: n_bins=%d", len(ref_counts))

    contingency_table = np.array([ref_counts, actual_counts])

    chi2, p_value, dof, _ = chi2_contingency(contingency_table)

    logger.debug(
        "chi2_p_value: chi2=%.6f, dof=%d, p_value=%.6f",
        chi2,
        dof,
        p_value,
    )

    return p_value


def cramer_v_measure_of_association(ref_counts, actual_counts):
    logger.debug(
        "cramer_v_measure_of_association: n_bins=%d",
        len(ref_counts),
    )

    contingency_table = np.array([ref_counts, actual_counts])
    cramer_v = association(contingency_table, method="cramer", correction=True)

    logger.debug("cramer_v_measure_of_association: value=%.6f", cramer_v)

    return cramer_v


def unseen_category_rate(ref_freq: dict, actual_freq: dict) -> float:
    logger.debug(
        "unseen_category_rate: ref_categories=%d, actual_categories=%d",
        len(ref_freq),
        len(actual_freq),
    )

    ref_cats = set(ref_freq.keys())

    unseen_count = sum(
        freq for cat, freq in actual_freq.items()
        if cat not in ref_cats
    )
    total_count = sum(actual_freq.values())

    if total_count == 0:
        logger.warning(
            "unseen_category_rate: total_count is zero, returning 0.0"
        )
        return 0.0

    result = unseen_count / total_count
    logger.debug(
        "unseen_category_rate: unseen_count=%d, total_count=%d, rate=%.6f",
        unseen_count,
        total_count,
        result,
    )

    return result


def category_churn(ref_freq: dict, actual_freq: dict) -> float:
    logger.debug(
        "category_churn: ref_categories=%d, actual_categories=%d",
        len(ref_freq),
        len(actual_freq),
    )

    ref_cats = set(ref_freq.keys())
    actual_cats = set(actual_freq.keys())

    new_cats = actual_cats - ref_cats          # появились
    disappeared_cats = ref_cats - actual_cats   # исчезли

    all_cats = ref_cats | actual_cats

    if len(all_cats) == 0:
        logger.warning("category_churn: no categories found in either dataset, returning 0.0")
        return 0.0

    result = (len(new_cats) + len(disappeared_cats)) / len(all_cats)
    logger.debug(
        "category_churn: new=%d, disappeared=%d, total_unique=%d, churn=%.6f",
        len(new_cats),
        len(disappeared_cats),
        len(all_cats),
        result,
    )

    return result


def cardinality_ratio_abs_diff(ref_cardinality_ratio: float, current: pd.Series) -> float:
    logger.debug(
        "cardinality_ratio_abs_diff: ref_ratio=%.6f, current_size=%d",
        ref_cardinality_ratio,
        len(current),
    )

    if len(current.dropna()) == 0:
        logger.warning(
            "cardinality_ratio_abs_diff: current series has no non-null values, returning 0"
        )
        return 0

    cur_cardinality_ratio = current.nunique() / len(current.dropna())

    result = abs(cur_cardinality_ratio - ref_cardinality_ratio)
    logger.debug(
        "cardinality_ratio_abs_diff: current_ratio=%.6f, abs_diff=%.6f",
        cur_cardinality_ratio,
        result,
    )

    return result
