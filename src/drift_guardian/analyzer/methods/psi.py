from drift_guardian.config_handler.parse_config import Metric
from drift_guardian.schema.models import ReferenceDict
from drift_guardian.analyzer.regestry.metric_registry import register
from drift_guardian.analyzer.methods.low_level.stats import psi
from drift_guardian.analyzer.utils import make_counts, find_ref

import pandas as pd


@register(Metric.psi)
def compute_psi(reference_dict: ReferenceDict, current: pd.Series) -> float:
    feature = current.name
    assert isinstance(feature, str)

    reference = find_ref(reference_dict, feature)
    ref_counts, cur_counts = make_counts(reference, current)
    return psi(ref_counts, cur_counts)



    
