from drift_guardian.config_handler.auto_config_builder import build_drift_config, ConfigBuildOptions
from drift_guardian.config_handler.parse_config import read_config, config_from_dict
from drift_guardian.profiler.baseline_profiler import Profiler
from drift_guardian.data_quality_checker.checker import SchemaChecker
from drift_guardian.analyzer.engine.engine import DriftMetricsEngine
from drift_guardian.analyzer.utils import extract_feature_groups

import logging
import warnings
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class OfflineWrapper:
    def __init__(self,
                 reference_df: pd.DataFrame,
                 path_to_config: str | None = None,
                 config_options: ConfigBuildOptions | None = None,
                 save_config_path: str | None = None,
                 merge_threshold: int = 5,
                 low_cardinality_threshold: int = 15,
                 take_sample: bool = False,
                 window_size: int | None = None
                 ):
        logger.debug(
            "Initializing OfflineWrapper: reference_df.shape=%s, "
            "path_to_config=%s, config_options=%s, save_config_path=%s, "
            "merge_threshold=%s, low_cardinality_threshold=%s",
            getattr(reference_df, "shape", None),
            path_to_config,
            config_options,
            save_config_path,
            merge_threshold,
            low_cardinality_threshold,
        )

        if path_to_config is not None and config_options is not None:
            logger.error(
                "Both path_to_config and config_options were provided. "
                "Initialization aborted."
            )
            raise ValueError("Provide path_to_config or config_options or nothing. Got both.")

        if path_to_config is not None:
            logger.info("Loading configuration from file: %s", path_to_config)
            self.config = read_config(path_to_config)
            if save_config_path:
                logger.warning(
                    "save_config_path=%s was provided together with path_to_config. "
                    "save_config_path will be ignored.",
                    save_config_path,
                )
                warnings.warn("save_config_path is provided with path_to_config. save_config_path will not be used")
        else:
            logger.info(
                "No configuration provided, auto-generating configuration from reference_df "
                "(save_config_path=%s)",
                save_config_path,
            )
            raw = build_drift_config(reference_df, options=config_options, output_path=save_config_path)
            self.config = config_from_dict(raw)
            logger.debug("Auto-generated configuration: %s", self.config)

        num_feats, cat_feats, prediction = extract_feature_groups(self.config)
        logger.debug(
            "Extracted feature groups: num_feats=%s, cat_feats=%s, prediction=%s",
            num_feats, cat_feats, prediction,
        )

        logger.info("Building reference data profile (Profiler)")
        profiler = Profiler(reference_df,
                            num_features=num_feats,
                            cat_features=cat_feats,
                            prediction=prediction,
                            merge_threshold=merge_threshold,
                            low_cardinality_threshold=low_cardinality_threshold,
                            take_sample=take_sample,
                            window_size=window_size)

        required_features = set(num_feats or []) | set(cat_feats or []) | {prediction}
        required_features.discard(None)
        logger.debug("Required features list: %s", required_features)

        self.required_features = required_features
        self.reference_dict = profiler.profile_ref_data()
        logger.info("Reference data profile built successfully")

        self.checker = SchemaChecker(reference_df, required_features)
        self.engine = DriftMetricsEngine(config=self.config, reference_profile=self.reference_dict)
        logger.info("OfflineWrapper initialized successfully")

    def analyze_df(self, current: pd.DataFrame):
        logger.info("Starting analyze_df for current.shape=%s", getattr(current, "shape", None))
        logger.debug("Validating current DataFrame schema via SchemaChecker")
        self.checker.check_df(current)

        logger.debug("Running DriftMetricsEngine.analyze_dataframe")
        report = self.engine.analyze_dataframe(current[list(self.required_features)])
        logger.info("analyze_df completed successfully")
        return report

    def run_av(self,
               current: pd.DataFrame,
               max_samples: int = 100_000,
               n_splits: int = 3,
               random_state: int = 42,
               missing_category: str = "__missing__",
               lightgbm_params: dict[str, Any] | None = None,
               ):
        logger.info(
            "Starting run_av: current.shape=%s, max_samples=%s, n_splits=%s, "
            "random_state=%s, missing_category=%s, lightgbm_params=%s",
            getattr(current, "shape", None),
            max_samples, n_splits, random_state, missing_category, lightgbm_params,
        )
        logger.debug("Validating current DataFrame schema via SchemaChecker")
        self.checker.check_df(current)

        logger.debug("Running DriftMetricsEngine.run_adversarial_validation")
        av_report = self.engine.run_adversarial_validation(current=current[list(self.required_features)],
                                                           max_samples=max_samples,
                                                           n_splits=n_splits,
                                                           random_state=random_state,
                                                           missing_category=missing_category,
                                                           lightgbm_params=lightgbm_params,
                                                           )
        logger.info("run_av completed successfully")
        return av_report
