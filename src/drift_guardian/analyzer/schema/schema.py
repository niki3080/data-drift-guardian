from typing import TypedDict, Literal, NotRequired, Protocol
import numpy as np
import pandas as pd

class OtherBucket(TypedDict):
    categories: list[str]
    is_catch_all_for_unseen: bool
    merge_cats_sum: int


class MergeInfo(TypedDict):
    merge_threshold: int
    other_bucket: OtherBucket


class Quantiles(TypedDict):
    p01: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

class CategoricalRef(TypedDict):
    feature: str
    type: Literal["categorical"]
    n: np.integer
    cardinality_ratio: np.floating
    missing_rate: np.floating
    categories: dict[str, int]
    proportions: dict[str, float]
    is_complete_category_list: bool
    merge_info: MergeInfo
    churn_baseline: Literal["reference"]


class NumericRef(TypedDict):
    type: Literal["numeric"]
    n: np.integer
    missing_rate: np.floating
    mean: float
    std: float
    max: float
    min: float
    quantiles: Quantiles
    low_cardinality: bool

    cardinality_ratio: NotRequired[np.floating]
    categories: NotRequired[dict]
    proportions: NotRequired[dict]
    is_complete_category_list: NotRequired[bool]
    merge_info: NotRequired[MergeInfo]
    churn_baseline: NotRequired[Literal["reference"]]

CatRef = dict[str, CategoricalRef]
PredsRef = dict[str, NumericRef | CategoricalRef]

class ReferenceDict(TypedDict):
    cat_ref: CatRef
    num_ref: NumericRef
    sample: pd.DataFrame
    preds_ref: PredsRef


class MetricFn(Protocol):
    def __call__(
        self,
        reference_dict: ReferenceDict,
        current: pd.Series,
        **kwargs,
    ) -> float: ...
