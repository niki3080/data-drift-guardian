import warnings
from typing import Hashable

import pandas as pd

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_period_dtype,
    is_numeric_dtype,
    is_bool_dtype,
    is_integer_dtype,
)

import numpy as np


class Profiler:
    def __init__(
        self,
        ref_data: pd.DataFrame,
        window_size: int,
        num_features: list | None = None,
        cat_features: list | None = None,
        prediction: str | None = None,
        merge_threshold: int = 5,
        low_cardinality_threshold: int = 15,
        take_sample: bool = True,
        sample_float_dtype="float32",
        random_state: int | None = None,
    ):
        if not isinstance(ref_data, pd.DataFrame):
            raise ValueError("ref_data must be provided in pandas DataFrame format")

        # Normalize feature lists
        num_features = list(num_features) if num_features is not None else []
        cat_features = list(cat_features) if cat_features is not None else []

        # Validate feature names
        if not all(isinstance(feature, str) for feature in num_features):
            raise ValueError("All num_features must be strings")

        if not all(isinstance(feature, str) for feature in cat_features):
            raise ValueError("All cat_features must be strings")

        # Check duplicate features
        if len(num_features) != len(set(num_features)):
            duplicates = sorted(
                {feature for feature in num_features if num_features.count(feature) > 1}
            )
            raise ValueError(f"Duplicate feature(s) in num_features: {duplicates}")

        if len(cat_features) != len(set(cat_features)):
            duplicates = sorted(
                {feature for feature in cat_features if cat_features.count(feature) > 1}
            )
            raise ValueError(f"Duplicate feature(s) in cat_features: {duplicates}")

        # A feature cannot be both numerical and categorical
        overlap = set(num_features) & set(cat_features)
        if overlap:
            raise ValueError(
                f"Feature(s) provided in both num_features and cat_features: "
                f"{sorted(overlap)}"
            )

        # Validate prediction
        if not isinstance(prediction, str | None):
            raise ValueError(f"prediction must be string or None, got {type(prediction)}")

        # Prediction cannot be explicitly specified in both feature groups
        if prediction is not None and prediction in num_features and prediction in cat_features:
            raise ValueError(
                f"Prediction '{prediction}' is provided in both num_features and cat_features. "
                "Choose one or leave prediction out of feature lists."
            )

        # Validate window_size
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError(f"window_size must be int, got {type(window_size)}")

        if window_size <= 0:
            raise ValueError(f"window_size must be greater than 0, got {window_size}")

        # Validate merge_threshold
        if not isinstance(merge_threshold, int) or isinstance(merge_threshold, bool):
            raise ValueError(
                f"merge_threshold must be int, got {type(merge_threshold)}"
            )

        if merge_threshold <= 0:
            raise ValueError(
                f"merge_threshold must be greater than 0, got {merge_threshold}"
            )

        # Validate low_cardinality_threshold
        if not isinstance(low_cardinality_threshold, int) or isinstance(
            low_cardinality_threshold, bool
        ):
            raise ValueError(
                "low_cardinality_threshold must be int, "
                f"got {type(low_cardinality_threshold)}"
            )

        if low_cardinality_threshold <= 0:
            raise ValueError(
                "low_cardinality_threshold must be greater than 0, "
                f"got {low_cardinality_threshold}"
            )

        # Validate take_sample
        if not isinstance(take_sample, bool):
            raise ValueError(f"sample must be bool, got {type(take_sample)}")

        # Validate sample dtype
        if not isinstance(sample_float_dtype, str):
            raise ValueError(
                f"sample_dtype must be str, got {type(sample_float_dtype)}"
            )

        try:
            sample_dtype_np = np.dtype(sample_float_dtype)
        except TypeError as exc:
            raise ValueError(
                f"Invalid numpy dtype for sample_dtype: {sample_float_dtype}"
            ) from exc

        if not np.issubdtype(sample_dtype_np, np.floating):
            raise ValueError(
                f"sample_dtype must be a numpy floating dtype, got {sample_float_dtype}"
            )

        # Validate random_state
        if not isinstance(random_state, int | None) or isinstance(random_state, bool):
            raise ValueError(
                f"random_state must be int or None, got {type(random_state)}"
            )

        # At least one thing must be profiled
        if not num_features and not cat_features and prediction is None:
            raise ValueError(
                "Something from num_features, cat_features or prediction "
                "must be provided. Nothing to profile."
            )

        self.window_size = window_size
        self.num_features = num_features
        self.cat_features = cat_features
        self.take_sample = take_sample
        self.sample_float_dtype = sample_float_dtype
        self.merge_threshold = merge_threshold
        self.low_cardinality_threshold = low_cardinality_threshold
        self.random_state = random_state

        columns_set = set(ref_data.columns)
        not_met_num_cols = []
        not_met_cat_cols = []

        for num_col in self.num_features:
            if num_col not in columns_set:
                not_met_num_cols.append(num_col)
                continue

            if time_dtype := self.check_for_time_dtype(ref_data, num_col):
                raise ValueError(
                    f"Num feature: {num_col} has dtype: {time_dtype} which is time dtype. Time dtypes are unsupported."
                )

            if not is_numeric_dtype(ref_data[num_col]):
                raise ValueError(
                    f"Col: {num_col} is mentioned in num_features, but doesn't have numeric dtype."
                )

            if is_bool_dtype(ref_data[num_col]):
                raise ValueError(
                    f"Col: {num_col} is mentioned in num_features, but has bool dtype."
                )

        for cat_col in self.cat_features:
            if cat_col not in columns_set:
                not_met_cat_cols.append(cat_col)
                continue

            if time_dtype := self.check_for_time_dtype(ref_data, cat_col):
                raise ValueError(
                    f"Cat feature: {cat_col} has dtype: {time_dtype} which is time dtype. Time dtypes are unsupported."
                )

        error_message = ""
        if prediction is not None and prediction not in columns_set:
            error_message = error_message + f"Prediction: {[prediction]} is missing in ref_data"

        if not_met_num_cols:
            if error_message:
                error_message = (
                    error_message
                    + ","
                    + " "
                    + f"num_feature(s): {not_met_num_cols} is missing in ref_data"
                )
            else:
                error_message = (
                    error_message
                    + f"num_feature(s): {not_met_num_cols} is missing in ref_data"
                )

        if not_met_cat_cols:
            if error_message:
                error_message = (
                    error_message
                    + ","
                    + " "
                    + f"cat_feature(s): {not_met_cat_cols} is missing in ref_data"
                )
            else:
                error_message = (
                    error_message
                    + f"cat_feature(s): {not_met_cat_cols} is missing in ref_data"
                )

        if error_message:
            raise ValueError(error_message)

        self.prediction = prediction
        if prediction is not None:
            if time_dtype := self.check_for_time_dtype(ref_data, self.prediction):
                raise ValueError(
                    f"Prediction col: {self.prediction} has dtype: {time_dtype} which is time dtype. Time dtypes are unsupported."
                )

            if prediction in self.num_features and not is_numeric_dtype(ref_data[prediction]):
                raise ValueError(
                    f"Prediction mentioned in num_features but doesn't have numeric dtype"
                )

            if prediction not in self.num_features and prediction not in self.cat_features:
                prediction_type = "num" if is_numeric_dtype(ref_data[prediction]) else "cat"
                if prediction_type == "num" and prediction not in self.num_features:
                    self.num_features.append(prediction)
                elif prediction_type == "cat" and prediction not in self.cat_features:
                    self.cat_features.append(prediction)

        self.ref_data = ref_data[
            num_features + cat_features
        ]  # prediction уже в одной из них

    def profile_ref_data(self):

        cat_ref = {}
        num_ref = {}

        prediction_ref = {}

        for col in self.cat_features:
            cat_result = self._profile_cat_feature(col)
            if col != self.prediction:
                cat_ref[col] = cat_result
            else:
                prediction_ref["type"] = "cat"
                prediction_ref[col] = cat_result
                prediction_ref["raw"] = self.ref_data[col]

        for col in self.num_features:
            thresh = self.low_cardinality_threshold
            ref_data = self.ref_data
            low_cardinality = ref_data[col].nunique() <= thresh

            if low_cardinality:
                num_result = self._profile_low_card_num_feature(col)
                num_result["low_cardinality"] = True
            else:
                num_result = self._profile_num_feature(col)
                num_result["low_cardinality"] = False

            if col != self.prediction:
                num_ref[col] = num_result
            else:
                prediction_ref["type"] = "num"
                prediction_ref[col] = num_result
                prediction_ref["raw"] = ref_data[col]

        if prediction_ref and (prediction_ref[self.prediction]["missing_rate"] > 0):
            missing_rate = prediction_ref[self.prediction]["missing_rate"]
            warnings.warn(
                f"Missing values in prediction. Missing rate: {missing_rate}", UserWarning
            )

        if self.take_sample:
            sample = self.build_reference_sample(
                ref_data, self.window_size, random_state=self.random_state
            )
            sample = self.compress_df(sample)
        else:
            sample = self.ref_data

        return cat_ref, num_ref, sample, prediction_ref

    @staticmethod
    def build_reference_sample(
        full_values: pd.DataFrame,
        window_size: int,
        min_size: int = 5000,
        multiplier: int = 10,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        target_size = max(min_size, multiplier * window_size)
        if len(full_values) > target_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(full_values), size=target_size, replace=False)
            full_values = full_values.iloc[idx]
        return full_values

    def compress_df(self, sample: pd.DataFrame) -> pd.DataFrame:
        sample_float_dtype = self.sample_float_dtype
        dtypes = {col: "category" for col in self.cat_features}

        for col in sample.select_dtypes(include="number").columns:
            s = sample[col]
            is_whole = is_integer_dtype(s) or (s.dropna() % 1 == 0).all()

            if is_whole:
                downcast = "unsigned" if s.min() >= 0 else "integer"
                dtypes[col] = pd.to_numeric(s, downcast=downcast).dtype
            else:
                dtypes[col] = sample_float_dtype

        return sample.astype(dtypes)

    def _profile_cat_feature(self, cat_feature):
        # Дополнительно в расчете PSI добавить сглаживание и cap на бакет для случая сумма < merge_threshold
        column = self.ref_data[cat_feature]

        if len(column) == 0:
            raise ValueError(f"Column '{cat_feature}' is empty")

        thresh = self.merge_threshold

        missing = column.isna().sum()

        if missing == len(column):
            raise ValueError(f"All values in column '{cat_feature}' is missing")

        missing_rate = missing / len(column)
        n_without_missing = len(column) - missing

        counts = column.value_counts()

        proportions: dict[Hashable, float] = (counts / n_without_missing).to_dict()
        frequencies: dict[Hashable, int] = counts.to_dict()

        cats_to_merge = counts[counts < thresh].index.to_list()
        merge_proportion = (counts[counts < thresh] / n_without_missing).sum()

        merge_info = {
            "merge_threshold": thresh,
            "other_bucket": {
                "categories": cats_to_merge,
                "is_catch_all_for_unseen": True,
                "proportion": merge_proportion,
            },
        }

        result = {
            "feature": cat_feature,
            "type": "categorical",
            "n": n_without_missing,
            "missing_rate": missing_rate,
            "categories": frequencies,
            "proportions": proportions,
            "is_complete_category_list": True,
            "merge_info": merge_info,
            "churn_baseline": "reference",
        }

        return result

    @staticmethod
    def check_for_time_dtype(ref_data, col):
        dtype = ref_data[col].dtype
        if is_datetime64_any_dtype(dtype):
            return dtype
        elif is_timedelta64_dtype(dtype):
            return dtype
        elif is_period_dtype(dtype):
            return dtype
        return False

    def _profile_num_feature(self, num_feature):
        column = self.ref_data[num_feature]

        if len(column) == 0:
            raise ValueError(f"Column '{num_feature}' is empty")

        missing = column.isna().sum()

        if missing == len(column):
            raise ValueError(f"All values in column '{num_feature}' is missing")

        missing_rate = missing / len(column)
        n_without_missing = len(column) - missing

        column_mean = column.mean()
        column_std = column.std()
        column_max = column.max()
        column_min = column.min()

        bounds_for_quantile_drift = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        new_axis = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]

        values_for_quantile_drift = (
            column.quantile(bounds_for_quantile_drift)
            .set_axis(new_axis, axis=0)
            .to_dict()
        )

        bounds_for_deciles = np.linspace(0, 1, 11, dtype="float32")
        deciles = column.quantile(bounds_for_deciles).to_list()
        deciles[0], deciles[-1] = -np.inf, np.inf
        frequencies, _ = np.histogram(column.dropna(), deciles)
        proportions_in_bins = frequencies / n_without_missing

        result = {
            "type": "numeric",
            "n": n_without_missing,
            "missing_rate": missing_rate,
            "mean": column_mean,
            "std": column_std,
            "max": column_max,
            "min": column_min,
            "quantiles": values_for_quantile_drift,
            "decile_bins": {
                "edges": deciles,
                "frequencies": frequencies,
                "proportions": proportions_in_bins,
            },
        }

        return result

    def _profile_low_card_num_feature(self, num_low_cord_feature):
        result_num = self._profile_num_feature(num_low_cord_feature)
        del result_num["decile_bins"]

        result_cat = self._profile_cat_feature(num_low_cord_feature)

        result_num["low_cardinality"] = True
        result_num["categories"] = result_cat["categories"]
        result_num["proportions"] = result_cat["proportions"]
        result_num["is_complete_category_list"] = result_cat[
            "is_complete_category_list"
        ]
        result_num["merge_info"] = result_cat["merge_info"]
        result_num["churn_baseline"] = result_cat["churn_baseline"]

        return result_num
