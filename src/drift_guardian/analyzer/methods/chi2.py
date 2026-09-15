from config.parse_config import Metric
from src.drift_guardian.analyzer.schema.schema import ReferenceDict
from src.drift_guardian.analyzer.regestry.registry import register
from src.drift_guardian.analyzer.methods.low_level.low_level import chi2_p_value
from src.drift_guardian.analyzer.utils.utils import make_counts, find_ref

import pandas as pd

@register(Metric.chi2)
def compute_chi2_p_value(reference_dict: ReferenceDict, current: pd.Series) -> float:
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)
    ref_counts, cur_counts = make_counts(reference, current)
    return chi2_p_value(ref_counts, cur_counts)


# @register(Metric.cramer_v)
# def compute_cramer_v(reference_dict: ReferenceDict, current: pd.Series) -> float:
#     feature = current.name
#     assert isinstance(feature, str)
#
#     reference = find_ref(reference_dict, feature)
#     ref_counts, cur_counts = make_counts(reference, current)
#     return psi(ref_counts, cur_counts)