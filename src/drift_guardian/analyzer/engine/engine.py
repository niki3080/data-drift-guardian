from src.drift_guardian.analyzer.regestry.registry import METRIC_REGISTRY
from src.drift_guardian.analyzer.schema.schema import ReferenceDict, MetricFn
from config.parse_config import Metric
from src.drift_guardian.analyzer.utils.utils import find_ref

from datetime import datetime, timezone

import pandas as pd

class DriftMetricsEngine:
    def __init__(self, config, reference_profile: ReferenceDict):
        self.config = config
        self.reference_dict = reference_profile

        self.window_size = None

    def analyze_dataframe(self, current: pd.DataFrame):
        self.window_size = len(current)

        config = self.config
        reference_dict = self.reference_dict

        results = {}

        for feature in config.features:
            feature_result = {}

            reference = find_ref(reference_dict, feature)

            feature_result['type'] = reference[feature]['type']

            for metric in config.features[feature].metrics:
                metrics_result = {}

                fn: MetricFn = METRIC_REGISTRY[metric]
                value: float = fn(reference_dict, current[feature]) #значение метрики
                thresh_warning = config.thresholds[metric].warning
                thresh_critical = config.thresholds[metric].critical

                lower_is_better = thresh_warning < thresh_critical
                metric_status = "ok"

                if lower_is_better:
                    if  thresh_critical > value > thresh_warning:
                        metric_status = "warning"
                    elif value > thresh_critical:
                        metric_status = "critical"
                else:
                    if thresh_critical < value < thresh_warning:
                        metric_status = "warning"
                    elif value < thresh_critical:
                        metric_status = "critical"

                metrics_result[metric.value] = {
                    "value": value,
                    "warning": thresh_warning,
                    "critical": thresh_critical,
                    "status": metric_status
                }
            feature_status = self._get_feature_status(metrics_result)

            feature_result['status'] = feature_status
            feature_result['metrics'] = metrics_result

            results[feature] = feature_result


        return self._make_report(results)

    @staticmethod
    def _get_feature_status(metrics_result: dict):
        feature_status = "ok"

        for metric in metrics_result.keys():
            metric_status = metric['status']
            if metric_status == 'critical' and metric != Metric.chi2:
                feature_status = 'critical'
                break
            if metric_status == 'warning':
                feature_status = 'warning'

        return feature_status

    def _make_report(self, monitoring_results: dict):
        overall_status, active_alerts = self._get_overall_status(monitoring_results)

        report = {'timestamp': self._get_current_timestamp(),
                  'window_size': self.window_size,
                  'overall_status': overall_status,
                  'features': monitoring_results}
        return report

    @staticmethod
    def _get_overall_status(monitoring_results: dict):
        overall_status = 'ok'
        active_alerts = 0

        for feature in monitoring_results.keys():
            feature_status = feature['status']

            if feature_status == 'critical':
                overall_status = 'critical'
                active_alerts += 1
            if overall_status != 'critical' and feature_status == 'warning':
                overall_status = 'warning'

        return overall_status, active_alerts

    @staticmethod
    def _get_current_timestamp():
        now_utc = datetime.now(timezone.utc)

        timestamp_str = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

        return timestamp_str