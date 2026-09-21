from src.drift_guardian.analyzer.regestry.metric_registry import METRIC_REGISTRY
from src.drift_guardian.analyzer.methods.batch.adversarial_validation import adversarial_validation
from src.drift_guardian.schema.models import ReferenceDict, MetricFn
from config.parse_config import Metric
from src.drift_guardian.analyzer.utils import find_ref

import logging
from typing import Any
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


class DriftMetricsEngine:
    def __init__(self, config, reference_profile: ReferenceDict):
        self.config = config
        self.reference_dict = reference_profile

        self.window_size = None

        logger.debug(
            "DriftMetricsEngine initialized with %d configured features",
            len(getattr(config, "features", {}) or {}),
        )

    def analyze_dataframe(self, current: pd.DataFrame):
        self.window_size = len(current)

        logger.info(
            "Starting drift analysis on current window (size=%d)",
            self.window_size,
        )

        config = self.config
        reference_dict = self.reference_dict

        feature_results = {}

        for feature in config.features:
            logger.debug("Analyzing feature column '%s'", feature)
            feature_result = self._analyze_column(reference_dict, current[feature], config)
            feature_results[feature] = feature_result

        prediction_result = None
        if config.prediction_metrics.enabled:
            prediction_column = config.prediction_metrics.score_column
            logger.debug("Analyzing prediction column '%s'", prediction_column)
            prediction_result = self._analyze_column(reference_dict, current[prediction_column], config, 'prediction')

        report = self._make_report(feature_results, prediction_result)

        logger.info(
            "Drift analysis completed: overall_status=%s, active_alerts=%d",
            report.get("overall_status"),
            report.get("active_alerts"),
        )

        return report

    def run_adversarial_validation(self,
                                   current: pd.DataFrame,
                                   max_samples: int = 100_000,
                                   n_splits: int = 3,
                                   random_state: int = 42,
                                   missing_category: str = "__missing__",
                                   lightgbm_params: dict[str, Any] | None = None,):

        logger.info(
            "Starting adversarial validation (max_samples=%d, n_splits=%d, random_state=%d)",
            max_samples, n_splits, random_state,
        )

        sample_df = self.reference_dict['sample']
        av_results = adversarial_validation(sample_df,
                                           current,
                                           max_samples=max_samples,
                                           n_splits=n_splits,
                                           random_state=random_state,
                                           missing_category=missing_category,
                                           lightgbm_params=lightgbm_params)

        logger.info("Adversarial validation completed")
        logger.debug("Adversarial validation results: %s", av_results)

        return av_results

    def _analyze_column(self, reference_dict: ReferenceDict, column_current: pd.Series, config, column_type='feature'):
            column_name = column_current.name
            assert isinstance(column_name, str)

            column_result = {}
            reference = find_ref(reference_dict, column_name)

            column_result['type'] = reference['type']
            metrics_result = {}


            if column_type == 'feature':
                metrics = config.features[column_name].metrics
                resolved_thresholds = config.features[column_name].resolved_thresholds
            elif column_type == 'prediction':
                metrics = config.prediction_metrics.metrics
                resolved_thresholds = config.prediction_metrics.resolved_thresholds
            else:
                logger.error("Invalid column_type '%s' provided for column '%s'", column_type, column_name)
                raise ValueError(f"column_type must be feature or prediction, got {column_type}")

            for metric in metrics:
                fn: MetricFn = METRIC_REGISTRY[metric]
                value: float = fn(reference_dict, column_current) #значение метрики
                threshold_pair = resolved_thresholds[metric]
                thresh_warning = threshold_pair.warning
                thresh_critical = threshold_pair.critical

                metric_status  = self._get_metric_status(thresh_warning, thresh_critical, value)

                if metric_status != "ok":
                    logger.warning(
                        "Column '%s': metric '%s' status=%s (value=%.4f, warning=%.4f, critical=%.4f)",
                        column_name, metric.value, metric_status, value, thresh_warning, thresh_critical,
                    )
                else:
                    logger.debug(
                        "Column '%s': metric '%s' status=ok (value=%.4f)",
                        column_name, metric.value, value,
                    )

                metrics_result[metric.value] = {
                    "value": value,
                    "warning": thresh_warning,
                    "critical": thresh_critical,
                    "status": metric_status
                }
            feature_status = self._get_feature_status(metrics_result)

            if feature_status != "ok":
                logger.warning("Column '%s' overall status=%s", column_name, feature_status)

            column_result['status'] = feature_status
            column_result['metrics'] = metrics_result

            return column_result

    @staticmethod
    def _get_metric_status(thresh_warning, thresh_critical, metric_value):
        lower_is_better = thresh_warning < thresh_critical
        metric_status = "ok"

        if lower_is_better:
            if  thresh_critical > metric_value > thresh_warning:
                metric_status = "warning"
            elif metric_value > thresh_critical:
                metric_status = "critical"
        else:
            if thresh_critical < metric_value < thresh_warning:
                metric_status = "warning"
            elif metric_value < thresh_critical:
                metric_status = "critical"

        return metric_status

    @staticmethod
    def _get_feature_status(metrics_result: dict):
        feature_status = "ok"

        for metric in metrics_result.keys():
            metric_status = metrics_result[metric]['status']
            if metric_status == 'critical' and metric != Metric.chi2:
                feature_status = 'critical'
                break
            if metric_status == 'warning':
                feature_status = 'warning'

        return feature_status

    def _make_report(self, monitoring_features_results: dict, monitoring_prediction_result: dict):
        overall_status, active_alerts = self._get_overall_status(monitoring_features_results)

        logger.debug(
            "Building report: overall_status=%s, active_alerts=%d, has_prediction=%s",
            overall_status, active_alerts, monitoring_prediction_result is not None,
        )

        if monitoring_prediction_result is not None:
            report = {'timestamp': self._get_current_timestamp(),
                      'window_size': self.window_size,
                      'overall_status': overall_status,
                      'active_alerts' : active_alerts,
                      'features': monitoring_features_results,
                      'prediction': monitoring_prediction_result}
        else:
            report = {'timestamp': self._get_current_timestamp(),
                      'window_size': self.window_size,
                      'overall_status': overall_status,
                      'active_alerts': active_alerts,
                      'features': monitoring_features_results}

        return report

    @staticmethod
    def _get_overall_status(monitoring_results: dict):
        overall_status = 'ok'
        active_alerts = 0

        for feature in monitoring_results.keys():
            feature_status = monitoring_results[feature]['status']

            if feature_status == 'critical':
                overall_status = 'critical'
                active_alerts += 1
                logger.warning("Feature '%s' triggered a critical alert", feature)
            if overall_status != 'critical' and feature_status == 'warning':
                overall_status = 'warning'

        return overall_status, active_alerts

    @staticmethod
    def _get_current_timestamp():
        now_utc = datetime.now(timezone.utc)

        timestamp_str = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

        return timestamp_str
