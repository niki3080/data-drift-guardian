from src.drift_guardian.schema.models import CategoricalRef, NumericRef

import pandas as pd
import numpy as np


def make_counts(reference: CategoricalRef | NumericRef, current: pd.Series):
    if reference['type'] == 'categorical' or reference['low_cardinality']:
        ref_buckets = reference['categories'].copy()

        other_bucket_cats = set(
            reference['merge_info']['other_bucket']['categories']
        )
        reference_cats = set(ref_buckets.keys())

        normal_cats = reference_cats - other_bucket_cats

        cur_value_counts = current.value_counts()

        cur_buckets = cur_value_counts[
            cur_value_counts.index.isin(normal_cats)
        ].to_dict()

        other_mask = (
            cur_value_counts.index.isin(other_bucket_cats)
            | ~cur_value_counts.index.isin(reference_cats)
        )

        if other_bucket_cats or (~cur_value_counts.index.isin(reference_cats)).any():
            cur_buckets['other'] = int(cur_value_counts[other_mask].sum())

        if other_bucket_cats:
            ref_buckets['other'] = reference['merge_info']['other_bucket']['merge_cats_sum']

        ref_buckets = {
            k: v for k, v in ref_buckets.items()
            if k not in other_bucket_cats
        }

        for key in ref_buckets:
            cur_buckets.setdefault(key, 0)

        ref_buckets = dict(sorted(ref_buckets.items()))
        cur_buckets = dict(sorted(cur_buckets.items()))

        ref_counts = list(ref_buckets.values())
        cur_counts = list(cur_buckets.values())

    elif reference['type'] == 'numeric':
        ref_counts = reference['decile_bins']['frequencies']

        deciles = reference['decile_bins']['deciles']
        cur_counts, _ = np.histogram(current.dropna(), deciles)

    else:
        raise ValueError(f"Unknown feature type in reference: {reference['type']}")

    return ref_counts, cur_counts