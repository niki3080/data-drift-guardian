from drift_guardian.schema.models import CategoricalRef, NumericRef, ReferenceDict

def find_ref(reference_dict: ReferenceDict, feature: str) -> CategoricalRef | NumericRef:
    if feature in reference_dict['num_ref']:
        reference = reference_dict['num_ref'][feature]
    elif feature in reference_dict['cat_ref']:
        reference = reference_dict['cat_ref'][feature]
    elif feature in reference_dict['preds_ref']:
        reference = reference_dict['preds_ref'][feature]
    else:
        raise ValueError(f"No feature: {feature} in reference_dict")

    return reference