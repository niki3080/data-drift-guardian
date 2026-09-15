from scipy.spatial import distance
from scipy.stats import chi2_contingency, kstest
from scipy.stats.contingency import association


import numpy as np

#реализовано
def psi(ref_counts, actual_counts, epsilon=1e-4, bin_cap=2.0):
    n_bins = len(ref_counts)
    
    expected_pct = np.array(ref_counts) / sum(ref_counts)
    actual_pct = np.array(actual_counts) / sum(actual_counts)
    
    expected_pct = (expected_pct + epsilon) / (1 + epsilon * n_bins)
    actual_pct = (actual_pct + epsilon) / (1 + epsilon * n_bins)
    
    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    
    psi_components = np.clip(psi_components, -bin_cap, bin_cap)
    
    return np.sum(psi_components)

#реализовано
def js_divergence(ref_counts, actual_counts, base=2):
    js_distance = distance.jensenshannon(ref_counts, actual_counts, base=base)
    return js_distance ** 2

def quantile_drift(quantile_actual, quantile_ref, ref_std):
    drift = (quantile_actual - quantile_ref) / ref_std
    return max(drift)

#реализовано
def ks_d_statistic(raw_ref, raw_actual):
    test_results = kstest(raw_ref, raw_actual)
    return test_results.statistic

#реализовано
def chi2_p_value(ref_counts, actual_counts):
    contingency_table = np.array([ref_counts, actual_counts])
    
    _, p_value, _, _ = chi2_contingency(contingency_table)
    
    return p_value

#реализовано
def cramer_v_measure_of_association(ref_counts, actual_counts):
    contingency_table = np.array([ref_counts, actual_counts])
    cramer_v = association(contingency_table, method="cramer", correction=True)
    return cramer_v

#реализовано
def unseen_category_rate(ref_freq: dict, actual_freq: dict) -> float:
    ref_cats = set(ref_freq.keys())
    
    unseen_count = sum(
        freq for cat, freq in actual_freq.items() 
        if cat not in ref_cats
    )
    total_count = sum(actual_freq.values())
    
    if total_count == 0:
        return 0.0
    
    return unseen_count / total_count

#реализовано
def category_churn(ref_freq: dict, actual_freq: dict) -> float:
    ref_cats = set(ref_freq.keys())
    actual_cats = set(actual_freq.keys())
    
    new_cats = actual_cats - ref_cats          # появились
    disappeared_cats = ref_cats - actual_cats   # исчезли
    
    all_cats = ref_cats | actual_cats
    
    if len(all_cats) == 0:
        return 0.0
    
    return (len(new_cats) + len(disappeared_cats)) / len(all_cats)



