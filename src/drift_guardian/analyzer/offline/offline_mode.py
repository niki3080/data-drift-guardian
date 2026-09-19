from config.auto_config_builder import build_drift_config, ConfigBuildOptions
from config.parse_config import read_config, config_from_dict
from src.drift_guardian.profiler.baseline_profiler import Profiler
from src.drift_guardian.data_quality_checker.checker import SchemaChecker
from src.drift_guardian.analyzer.engine.engine import DriftMetricsEngine
from src.drift_guardian.analyzer.utils import extract_feature_groups

import warnings
from typing import Any

import pandas as pd

class OfflineWrapper:
    def __init__(self,
                 reference_df: pd.DataFrame,
                 path_to_config: str | None = None,
                 config_options: ConfigBuildOptions | None = None,
                 save_config_path: str | None = None,
                 merge_threshold: int = 5,
                 low_cardinality_threshold: int = 15,
                 ):

        if path_to_config is not None and config_options is not None:
            raise ValueError("Provide path_to_config or config_options or nothing. Got both.")


        if path_to_config is not None:
            self.config = read_config(path_to_config)
            if save_config_path:
                warnings.warn("save_config_path is provided with path_to_config. save_config_path will not be used")
        else:
            raw = build_drift_config(reference_df, options=config_options, output_path=save_config_path)
            self.config = config_from_dict(raw)


        num_feats, cat_feats, prediction = extract_feature_groups(self.config)

        profiler = Profiler(reference_df,
                            num_features=num_feats,
                            cat_features=cat_feats,
                            prediction=prediction,
                            merge_threshold=merge_threshold,
                            low_cardinality_threshold=low_cardinality_threshold,
                            take_sample=False)

        required_features = set(num_feats or []) | set(cat_feats or []) | {prediction}
        required_features.discard(None)

        self.required_features = required_features
        self.reference_dict = profiler.profile_ref_data()
        self.checker = SchemaChecker(reference_df, required_features)
        self.engine = DriftMetricsEngine(config=self.config, reference_profile=self.reference_dict)

    def analyze_df(self, current: pd.DataFrame):
        self.checker.check_df(current)

        report = self.engine.analyze_dataframe(current[list(self.required_features)])
        return report

    def run_av(self,
               current: pd.DataFrame,
               max_samples: int = 100_000,
               n_splits: int = 3,
               random_state: int = 42,
               missing_category: str = "__missing__",
               lightgbm_params: dict[str, Any] | None = None,
               ):
        self.checker.check_df(current)
        av_report = self.engine.run_adversarial_validation(current=current[list(self.required_features)],
                                                           max_samples=max_samples,
                                                           n_splits=n_splits,
                                                           random_state=random_state,
                                                           missing_category=missing_category,
                                                           lightgbm_params=lightgbm_params,
                                                           )
        return av_report