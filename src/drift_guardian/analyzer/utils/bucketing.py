from drift_guardian.schema.models import CategoricalRef, NumericRef

import numpy as np
import pandas as pd

OTHER_BUCKET = "other"

def make_counts(reference: CategoricalRef | NumericRef, current: pd.Series):
    feature_type = reference["type"]

    if feature_type == "numeric" and not reference.get("low_cardinality", False):
        bins = reference["decile_bins"]["deciles"]
        ref_counts = reference["decile_bins"]["frequencies"]

        cur_counts, _ = np.histogram(current.dropna(), bins=bins)

        return ref_counts, cur_counts

    if feature_type == "categorical" or (
        feature_type == "numeric" and reference.get("low_cardinality", False)
    ):
        categories = reference["categories"]

        merge_info = reference.get("merge_info") or {}
        other_bucket = merge_info.get("other_bucket") or {}

        merged_categories = set(other_bucket.get("categories") or [])
        other_ref_count = other_bucket.get("merge_cats_sum", 0)

        current_counts = current.value_counts(dropna=True).to_dict()

        ref_counts = []
        cur_counts = []

        known_non_merged_categories = []

        for category, ref_count in categories.items():
            if category in merged_categories:
                continue

            known_non_merged_categories.append(category)

            ref_counts.append(ref_count)
            cur_counts.append(current_counts.get(category, 0))

        known_categories = set(categories.keys())

        other_current_count = 0

        for category, count in current_counts.items():
            if category in merged_categories:
                other_current_count += count
            elif category not in known_categories:
                other_current_count += count

        has_other_bucket = bool(merged_categories) or other_ref_count > 0 or other_current_count > 0

        if has_other_bucket:
            ref_counts.append(other_ref_count)
            cur_counts.append(other_current_count)

        return ref_counts, cur_counts

    raise ValueError(f"Unknown feature type in reference: {feature_type}")