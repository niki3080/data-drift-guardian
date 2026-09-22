from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.core.parse_config import (
    Config,
    Metric,
    StreamDriftConfig,
    read_config,
)
from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.stream_metrics import load_stream_thresholds

ROOT = Path(__file__).resolve().parents[1]


def _dashboard_prometheus_metrics(dashboard: dict) -> set[str]:
    metrics: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, dict):
            expression = value.get("expr")
            if isinstance(expression, str):
                metrics.update(
                    re.findall(r"\bdrift_[A-Za-z0-9_]+", expression)
                )
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(dashboard)
    return metrics


def test_repository_uses_yaml_extension_consistently() -> None:
    yml_files = [path for path in ROOT.rglob("*.yml") if ".venv" not in path.parts]
    assert yml_files == []

    text_files = [
        ROOT / ".gitignore",
        ROOT / "README.md",
        ROOT / "docker-compose.yaml",
        ROOT / "monitoring/prometheus/prometheus.yaml",
    ]
    for path in text_files:
        assert '.yml' not in path.read_text(encoding='utf-8')

def test_single_compose_serves_realtime_and_mock_profiles() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]

    assert services['analyzer']['profiles'] == ['realtime']
    assert services['kafka']['profiles'] == ['realtime']
    assert services['kafka-init']['profiles'] == ['realtime']
    assert services['drift-producer']['profiles'] == ['realtime']
    assert services['drift-mock-exporter']['profiles'] == ['mock']

    producer_dependencies = services["drift-producer"]["depends_on"]
    assert producer_dependencies['analyzer']['condition'] == 'service_healthy'
    analyzer_environment = services["analyzer"]["environment"]
    assert not any((str(name).startswith('STREAM_') for name in analyzer_environment))

    analyzer_dockerfile = (
        ROOT / services["analyzer"]["build"]["dockerfile"]
    ).read_text(encoding="utf-8")
    assert 'libgomp1' in analyzer_dockerfile

    prometheus_mounts = services["prometheus"]["volumes"]
    assert './monitoring/prometheus/prometheus.yaml:/etc/prometheus/prometheus.yaml:ro' in prometheus_mounts

    assert services['prometheus']['image'] == 'prom/prometheus:v3.14.0'
    assert services['grafana']['image'] == 'grafana/grafana:13.2.1'

    grafana_mounts = services["grafana"]["volumes"]
    assert './monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro' in grafana_mounts
    assert './monitoring/grafana/provisioning:/etc/grafana/provisioning:ro' in grafana_mounts

def test_single_prometheus_config_targets_shared_exporter_alias() -> None:
    config = yaml.safe_load(
        (ROOT / "monitoring/prometheus/prometheus.yaml").read_text(encoding="utf-8")
    )
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    targets = jobs["drift-exporter"]["static_configs"][0]["targets"]
    assert targets == ['drift-exporter:8000']

def test_dashboard_is_provisioned_with_matching_datasource() -> None:
    dashboard_provider = yaml.safe_load(
        (
            ROOT
            / "monitoring/grafana/provisioning/dashboards/dashboards.yaml"
        ).read_text(encoding="utf-8")
    )
    provider = dashboard_provider["providers"][0]
    assert provider['options']['path'] == '/var/lib/grafana/dashboards'

    datasource = yaml.safe_load(
        (
            ROOT
            / "monitoring/grafana/provisioning/datasources/prometheus.yaml"
        ).read_text(encoding="utf-8")
    )["datasources"][0]
    assert datasource['url'] == 'http://prometheus:9090'

    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    assert dashboard['spec']['title'] == 'Data Drift Guardian v2.3'
    assert len(dashboard['spec']['elements']) == 33
    assert dashboard['apiVersion'] == 'dashboard.grafana.app/v2'
    assert dashboard['kind'] == 'Dashboard'
    assert 'Grafana v13.2.1' in dashboard['metadata']['annotations']['grafana.app/saved-from-ui']

    # dashboard должен ссылаться на UID datasource из provisioning
    dashboard_text = json.dumps(dashboard, ensure_ascii=False)
    assert str(datasource['uid']) in dashboard_text

def test_dashboard_history_defaults_and_queries_are_time_series() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )

    time_settings = dashboard["spec"]["timeSettings"]
    assert time_settings['from'] == 'now-30m'
    assert time_settings['to'] == 'now'
    assert time_settings['autoRefresh'] == '5s'

    for panel_id in ("panel-75", "panel-76", "panel-77", "panel-90"):
        query = (
            dashboard["spec"]["elements"][panel_id]["spec"]
            ["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]
        )
        assert query['format'] == 'time_series'
        assert not query['instant']
        assert query['range']

    for panel_id in ("panel-76", "panel-77"):
        expression = (
            dashboard["spec"]["elements"][panel_id]["spec"]
            ["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]["expr"]
        )
        assert 'feature=~"$history_feature"' in expression
        assert 'metric=~"$history_metric"' in expression
        assert ':regex' not in expression

    history_metric = None

    def find_history_metric(value) -> None:
        nonlocal history_metric
        if isinstance(value, dict):
            if value.get("name") == "history_metric":
                history_metric = value
                return
            for nested in value.values():
                find_history_metric(nested)
        elif isinstance(value, list):
            for nested in value:
                find_history_metric(nested)

    find_history_metric(dashboard)
    assert history_metric is not None
    assert history_metric['allValue'] == '.*'
    assert history_metric['query']['spec']['qryType'] == 1

def test_exporter_registers_every_metric_used_by_dashboard() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    dashboard_metrics = _dashboard_prometheus_metrics(dashboard)

    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)
    exporter.set_window_size(1000)
    exposition = generate_latest(registry).decode("utf-8")
    assert (
        "# HELP drift_av_reference_rows Reference rows available "
        "to adversarial validation"
    ) in exposition
    registered_metrics = set(
        re.findall(
            r"^# HELP ([A-Za-z_:][A-Za-z0-9_:]*) ",
            exposition,
            re.M,
        )
    )

    expected_metrics = {
        "drift_active_alerts",
        "drift_analysis_runs_total",
        "drift_av_driver_consistency",
        "drift_av_driver_similarity_previous",
        "drift_av_feature_importance",
        "drift_av_last_run_timestamp_seconds",
        "drift_av_roc_auc",
        "drift_av_roc_auc_cv_min",
        "drift_av_roc_auc_cv_std",
        "drift_av_status",
        "drift_av_reference_rows",
        "drift_av_current_rows",
        "drift_av_dataset_size",
        "drift_av_features_evaluated",
        "drift_av_sample_fraction",
        "drift_av_threshold",
        "drift_current_window_events",
        "drift_event_time_lag_seconds",
        "drift_events_processed_total",
        "drift_invalid_event_time_rate",
        "drift_late_events_total",
        "drift_max_event_gap_seconds",
        "drift_metric_value",
        "drift_out_of_order_events_total",
        "drift_overall_status",
        "drift_reference_profile_info",
        "drift_report_timestamp_seconds",
        "drift_status",
        "drift_status_feature",
        "drift_stream_status",
        "drift_threshold",
        "drift_window_size",
        "drift_window_time_span_seconds",
    }
    assert dashboard_metrics == expected_metrics
    assert dashboard_metrics - registered_metrics == set()


def test_dashboard_uses_direct_window_metric() -> None:
    dashboard_path = (
        ROOT / "monitoring/grafana/dashboards/drift_guardian.json"
    )
    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    dashboard = json.loads(dashboard_text)
    elements = dashboard["spec"]["elements"]

    assert "drift_current_window_events" in dashboard_text
    assert "drift_late_event_rate" not in dashboard_text
    assert "drift_events_processed_total % drift_window_size" not in dashboard_text
    assert "# approximation" not in dashboard_text
    assert (
        "floor(drift_events_processed_total / drift_window_size)"
        not in dashboard_text
    )
    assert "sum(round(increase" not in dashboard_text

    for metric in (
        "drift_events_processed_total",
        "drift_analysis_runs_total",
        "drift_out_of_order_events_total",
        "drift_late_events_total",
    ):
        assert f"round(sum(increase({metric}[$__range])))" in dashboard_text

    readiness_queries = {
        query["spec"]["query"]["spec"]["legendFormat"]: query["spec"]
        ["query"]["spec"]["expr"]
        for query in elements["panel-27"]["spec"]["data"]["spec"]["queries"]
    }
    assert readiness_queries["calculated_window_fill"].startswith(
        "max(drift_current_window_events)"
    )
    assert "max(drift_current_window_events)" in readiness_queries[
        "calculated_window_fill_percent"
    ]
    assert "max(drift_current_window_events)" in readiness_queries[
        "calculated_ETA"
    ]

    quality_panel = elements["panel-110"]["spec"]
    quality_queries = {
        query["spec"]["query"]["spec"]["legendFormat"]: query["spec"]
        ["query"]["spec"]["expr"]
        for query in quality_panel["data"]["spec"]["queries"]
    }
    assert quality_queries == {
        "invalid_event_time_rate": (
            "drift_invalid_event_time_rate or on() vector(-999)"
        ),
    }
    quality_canvas = json.dumps(quality_panel["vizConfig"], ensure_ascii=False)
    assert '"field": "late_event_rate"' not in quality_canvas
    assert '"fixed": "Late event rate"' not in quality_canvas
    quality_elements = quality_panel["vizConfig"]["spec"]["options"]["root"][
        "elements"
    ]
    invalid_label = next(
        element
        for element in quality_elements
        if element["name"] == "invalid_event_time_rate label"
    )
    assert invalid_label["config"]["size"] == 12

    since_start_queries = elements["panel-58"]["spec"]["data"]["spec"][
        "queries"
    ]
    assert [
        query["spec"]["query"]["spec"]["legendFormat"]
        for query in since_start_queries
    ] == [
        "events_processed_total",
        "analysis_runs",
        "out_of_order_events_total",
        "late_events",
    ]
    assert "process_start_time" not in {
        query["spec"]["query"]["spec"]["legendFormat"]
        for query in since_start_queries
    }


def test_dashboard_process_start_is_scoped_to_drift_exporter() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    rows = dashboard["spec"]["layout"]["spec"]["rows"]
    stream_row = next(
        row["spec"]
        for row in rows
        if any(
            variable["spec"]["name"] == "process_start_time_s"
            for variable in row["spec"].get("variables", [])
        )
    )
    process_start = next(
        variable["spec"]
        for variable in stream_row["variables"]
        if variable["spec"]["name"] == "process_start_time_s"
    )
    expected_query = (
        'query_result(max(process_start_time_seconds{job="drift-exporter"}) '
        "* 1000)"
    )

    assert process_start["query"]["spec"]["query"] == expected_query
    assert process_start["definition"] == expected_query


def test_dashboard_contains_conditional_av_row_with_agreed_metrics() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    rows = dashboard["spec"]["layout"]["spec"]["rows"]
    av_row = next(
        row for row in rows
        if row.get("spec", {}).get("title", "").startswith(
            "Adversarial Validation"
        )
    )
    conditional = av_row["spec"]["conditionalRendering"]["spec"]
    assert conditional['visibility'] == 'show'
    assert conditional['items'][0]['spec']['variable'] == 'av_exists'

    dashboard_text = json.dumps(dashboard, ensure_ascii=False)
    for metric in (
        "drift_av_available",
        "drift_av_last_run_timestamp_seconds",
        "drift_av_dataset_size",
        "drift_av_status",
        "drift_av_roc_auc",
        "drift_av_roc_auc_cv_min",
        "drift_av_roc_auc_cv_std",
        "drift_av_driver_consistency",
        "drift_av_driver_similarity_previous",
        "drift_av_feature_importance",
    ):
        assert metric in dashboard_text

def test_av_dashboard_matches_main_dashboard_layout() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    elements = dashboard["spec"]["elements"]

    summary = elements["panel-92"]["spec"]
    assert summary['vizConfig']['group'] == 'canvas'
    root = summary["vizConfig"]["spec"]["options"]["root"]
    assert root['background']['color']['field'] == 'av_status'
    assert not summary['vizConfig']['spec']['options']['inlineEditing']
    summary_text = json.dumps(summary, ensure_ascii=False)
    for field in ("av_status", "av_status_details", "av_last_run"):
        assert field in summary_text
    assert 'No meaningful shift' in summary_text
    assert 'Reference and current are similar' not in summary_text

    for panel_name, metric in (
        ("panel-101", "drift_av_roc_auc"),
        ("panel-98", "drift_av_roc_auc_cv_min"),
        ("panel-96", "drift_av_roc_auc_cv_std"),
        ("panel-99", "drift_av_driver_consistency"),
        ("panel-97", "drift_av_driver_similarity_previous"),
    ):
        assert metric in json.dumps(elements[panel_name], ensure_ascii=False)
    assert 'drift_av_dataset_size' not in json.dumps(elements['panel-98'], ensure_ascii=False)
    panel100_text = json.dumps(elements["panel-100"], ensure_ascii=False)
    for metric in (
        "drift_av_reference_rows",
        "drift_av_current_rows",
        "drift_av_dataset_size",
        "drift_av_features_evaluated",
        "drift_av_sample_fraction",
        "drift_av_threshold",
    ):
        assert metric in panel100_text
    panel97_text = json.dumps(elements["panel-97"], ensure_ascii=False)
    assert 'Prev-window sim.' in panel97_text
    assert 'Top-3 share' not in panel97_text
    assert 'CV std \u2193' not in json.dumps(elements['panel-96'], ensure_ascii=False)
    assert 'Driver stability \u2191' not in json.dumps(elements['panel-99'], ensure_ascii=False)
    assert 'Driver consistency' in json.dumps(elements['panel-99'], ensure_ascii=False)

    importance = elements["panel-93"]["spec"]
    importance_text = json.dumps(importance, ensure_ascii=False)
    importance_override = next(
        item
        for item in importance["vizConfig"]["spec"]["fieldConfig"]["overrides"]
        if item["matcher"].get("options") == "Importance"
    )
    importance_props = {
        item["id"]: item.get("value")
        for item in importance_override["properties"]
    }
    assert importance_props['unit'] == 'percentunit'
    assert importance_props['decimals'] == 1
    organize = next(
        item
        for item in importance["data"]["spec"]["transformations"]
        if item.get("group") == "organize"
    )
    assert organize['spec']['options']['excludeByName']['__name__']

    history = elements["panel-94"]["spec"]
    history_text = json.dumps(history, ensure_ascii=False)
    assert history['vizConfig']['group'] == 'status-history'
    assert 'drift_av_status' in history_text
    assert 'drift_av_roc_auc' not in history_text
    assert history['vizConfig']['spec']['options']['legend']['showLegend']
    status_mapping = history["vizConfig"]["spec"]["fieldConfig"]["defaults"]["mappings"][0]["options"]
    assert status_mapping['0']['text'] == 'Healthy'
    assert status_mapping['1']['text'] == 'Warning'
    assert status_mapping['2']['text'] == 'Critical'

    rows = dashboard["spec"]["layout"]["spec"]["rows"]
    av_row = next(
        row for row in rows
        if row.get("spec", {}).get("title", "").startswith(
            "Adversarial Validation"
        )
    )
    nested_rows = av_row["spec"]["layout"]["spec"]["rows"]
    latest_items = nested_rows[0]["spec"]["layout"]["spec"]["items"]
    latest_names = [item["spec"]["element"]["name"] for item in latest_items]
    assert latest_names == [
        'panel-92',
        'panel-101',
        'panel-98',
        'panel-96',
        'panel-99',
        'panel-97',
    ]
    assert len(nested_rows[1]['spec']['layout']['spec']['items']) == 2
    assert nested_rows[1]['spec']['layout']['spec']['items'][0]['spec']['height'] == 9
    assert nested_rows[2]['spec']['title'] == 'AV Status Over Time'
    assert nested_rows[0]['spec']['layout']['kind'] == 'GridLayout'
    top_items = nested_rows[0]["spec"]["layout"]["spec"]["items"]
    assert len(top_items) == 6
    assert top_items[0]['spec']['width'] == 5
    assert top_items[-2]['spec']['width'] == 4
    assert top_items[-1]['spec']['width'] == 4
    assert nested_rows[1]['spec']['layout']['spec']['items'][0]['spec']['width'] == 12
    assert nested_rows[1]['spec']['layout']['spec']['items'][1]['spec']['width'] == 12
    assert nested_rows[2]['spec']['layout']['spec']['items'][0]['spec']['width'] == 24

def test_status_timelines_preserve_all_status_codes_and_av_samples() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    elements = dashboard["spec"]["elements"]

    stream_history = elements["panel-91"]["spec"]
    stream_override = next(
        override
        for override in stream_history["vizConfig"]["spec"]["fieldConfig"][
            "overrides"
        ]
        if override["matcher"].get("options") == "Stream status"
    )
    stream_mapping = next(
        prop["value"][0]["options"]
        for prop in stream_override["properties"]
        if prop["id"] == "mappings"
    )
    assert stream_mapping['-1']['text'] == 'Not ready'
    assert stream_mapping['-1']['color'] == '#ccccdb80'
    assert stream_mapping['0']['text'] == 'Healthy'
    assert stream_mapping['1']['text'] == 'Degraded'
    assert stream_mapping['2']['text'] == 'Unhealthy'
    assert '3' not in stream_mapping

    av_history = elements["panel-94"]["spec"]
    av_query = av_history["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]
    assert av_query['expr'] == 'max(drift_av_status)'
    assert av_query['range']
    assert not av_query['instant']
    assert 'max_over_time' not in av_query['expr']
    av_mapping = (
        av_history["vizConfig"]["spec"]["fieldConfig"]["defaults"]
        ["mappings"][0]["options"]
    )
    assert av_mapping['0']['text'] == 'Healthy'
    assert av_mapping['1']['text'] == 'Warning'
    assert av_mapping['2']['text'] == 'Critical'

def test_av_quality_kpi_labels_are_centered_over_values() -> None:
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    elements = dashboard["spec"]["elements"]
    for panel_name in ("panel-99", "panel-97"):
        root = elements[panel_name]["spec"]["vizConfig"]["spec"]["options"]["root"]
        label = next(item for item in root["elements"] if item["type"] == "text")
        value = next(item for item in root["elements"] if item["type"] == "metric-value")
        assert label['config']['align'] == 'center'
        assert value['config']['align'] == 'center'
        assert label['placement']['left'] == value['placement']['left']
        assert label['placement']['width'] == value['placement']['width']

def test_threshold_contract_is_present_in_runtime_config() -> None:
    config = yaml.safe_load(
        (ROOT / "config/config.yaml").read_text(encoding="utf-8")
    )
    assert config['features']['age']['thresholds']['psi'] == {'warning': 0.05, 'critical': 0.12}
    assert config['thresholds']['missing_rate'] == {'warning': 0.02, 'critical': 0.05}
    assert config['stream_drift'] == {
        'drift_event_time_lag_seconds': {'warning': 30, 'critical': 120},
        'drift_late_event_rate': {'warning': 0.01, 'critical': 0.05},
    }
    assert config['adversarial_validation'] == {'enabled': True, 'thresholds': {'warning': 0.6, 'critical': 0.75}}

    parsed = read_config(ROOT / "config/config.yaml")
    assert parsed.adversarial_validation is not None
    assert parsed.adversarial_validation.thresholds.warning == 0.6
    assert parsed.adversarial_validation.thresholds.critical == 0.75
    assert parsed.features['age'].resolved_thresholds[Metric.psi].warning == 0.05
    assert parsed.features['income'].resolved_thresholds[Metric.psi].warning == 0.1
    assert parsed.stream_drift.drift_late_event_rate.critical == 0.05

    stream_thresholds = load_stream_thresholds(ROOT / "config/config.yaml")
    assert set(stream_thresholds.as_dict()) == {
        'drift_event_time_lag_seconds',
        'drift_late_event_rate',
    }

    with pytest.raises(ValueError, match="drift_late_events_total"):
        StreamDriftConfig(
            drift_late_events_total={"warning": 10, "critical": 50}
        )

def test_global_threshold_is_not_overwritten_by_feature_override() -> None:
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)
    exporter.update_report(
        {
            "overall_status": "warning",
            "active_alerts": 1,
            "window_size": 100,
            "thresholds": {
                "psi": {"warning": 0.1, "critical": 0.25},
            },
            "features": {
                "age": {
                    "type": "numeric",
                    "status": "warning",
                    "metrics": {
                        "psi": {
                            "value": 0.11,
                            "warning": 0.05,
                            "critical": 0.12,
                            "status": "warning",
                        }
                    },
                }
            },
        }
    )

    exposition = generate_latest(registry).decode("utf-8")
    assert 'drift_threshold{level="warning",metric="psi"} 0.1' in exposition
    assert 'drift_threshold{level="critical",metric="psi"} 0.25' in exposition
    assert 'drift_threshold{level="warning",metric="psi"} 0.05' not in exposition

def test_chi2_thresholds_use_reversed_p_value_direction() -> None:
    valid = {
        "features": {
            "country": {
                "type": "categorical",
                "metrics": ["chi2"],
            }
        },
        "prediction_metrics": {"enabled": False},
        "thresholds": {
            "chi2": {"warning": 0.05, "critical": 0.01},
        },
    }
    parsed = Config(**valid)
    assert parsed.features['country'].resolved_thresholds[Metric.chi2].warning == 0.05

    invalid = {
        **valid,
        "thresholds": {
            "chi2": {"warning": 0.01, "critical": 0.05},
        },
    }
    with pytest.raises(ValueError):
        Config(**invalid)

def test_mock_has_many_features_mixed_statuses_and_window_variation() -> None:
    script = r"""
import importlib.util
import json
from pathlib import Path
from prometheus_client import REGISTRY, generate_latest

module_path = Path("monitoring/mock_exporter/app.py")
spec = importlib.util.spec_from_file_location("mock_exporter_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.set_critical_scenario()
first = generate_latest(REGISTRY).decode("utf-8")
module.set_critical_scenario()
second = generate_latest(REGISTRY).decode("utf-8")
module.set_critical_scenario()
third = generate_latest(REGISTRY).decode("utf-8")
print(json.dumps([first, second, third]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    first, second, third = json.loads(completed.stdout)

    metric_pattern = re.compile(
        r'^drift_metric_value\{feature="([^"]+)",metric="([^"]+)",type="([^"]+)"\} ([^\n]+)$',
        re.M,
    )
    status_pattern = re.compile(
        r'^drift_status\{feature="[^"]+",metric="[^"]+",type="[^"]+"\} (-?\d+(?:\.\d+)?)$',
        re.M,
    )

    first_values = {
        (feature, metric, feature_type): float(value)
        for feature, metric, feature_type, value in metric_pattern.findall(first)
    }
    second_values = {
        (feature, metric, feature_type): float(value)
        for feature, metric, feature_type, value in metric_pattern.findall(second)
    }
    feature_names = {feature for feature, _, _ in first_values}
    assert len(feature_names - {'prediction'}) >= 15

    statuses = {float(value) for value in status_pattern.findall(first)}
    assert {0.0, 1.0, 2.0}.issubset(statuses)

    common = set(first_values) & set(second_values)
    changed = sum(first_values[key] != second_values[key] for key in common)
    assert changed > len(common) // 2

    assert 'drift_threshold{level="warning",metric="chi2"} 0.05' in first
    assert 'drift_threshold{level="critical",metric="chi2"} 0.01' in first
    assert 'drift_threshold{level="warning",metric="missing_rate"} 0.02' in first
    assert 'drift_threshold{level="critical",metric="cardinality_ratio"} 0.6' in first

    chi2_statuses = {
        float(value)
        for value in re.findall(
            r'^drift_status\{feature="[^"]+",metric="chi2",type="categorical"\} (-?\d+(?:\.\d+)?)$',
            first,
            re.M,
        )
    }
    assert {1.0, 2.0}.issubset(chi2_statuses)

    assert 'drift_av_available 1.0' in first
    assert 'drift_av_status 0.0' in first
    assert 'drift_av_status 1.0' in second
    assert 'drift_av_status 2.0' in third
    assert 'drift_av_roc_auc 0.54' in first
    assert 'drift_av_roc_auc 0.66' in second
    assert 'drift_av_roc_auc 0.8' in third
    assert 'drift_av_roc_auc_cv_std 0.008' in first
    assert 'drift_av_roc_auc_cv_std 0.018' in second
    assert 'drift_av_roc_auc_cv_std 0.032' in third
    assert 'drift_av_roc_auc_cv_min 0.529' in first
    assert 'drift_av_roc_auc_cv_min 0.638' in second
    assert 'drift_av_roc_auc_cv_min 0.761' in third
    assert 'drift_av_driver_consistency 0.94' in first
    assert 'drift_av_driver_consistency 0.86' in second
    assert 'drift_av_driver_consistency 0.72' in third
    assert 'drift_av_top1_importance_share 0.29' in first
    assert 'drift_av_top3_importance_share 0.64' in first
    assert 'drift_av_dataset_size 1000.0' in first
    assert "drift_current_window_events 0.0" in first
    assert "drift_late_event_rate 0.0" in first
    assert (
        'drift_stream_threshold{level="warning",'
        'metric="drift_late_event_rate"} 0.01'
        in first
    )


def test_mock_late_counter_and_window_rate_use_the_same_events() -> None:
    script = r"""
import importlib.util
import json
import random
import re
from pathlib import Path
from prometheus_client import REGISTRY, generate_latest

module_path = Path("monitoring/mock_exporter/app.py")
spec = importlib.util.spec_from_file_location("mock_stream_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = random.Random(0)

def snapshot():
    exposition = generate_latest(REGISTRY).decode("utf-8")
    def value(metric):
        match = re.search(rf"^{metric} ([^\\n]+)$", exposition, re.M)
        return float(match.group(1))
    return {
        "late_total": value("drift_late_events_total"),
        "late_rate": value("drift_late_event_rate"),
        "out_of_order_total": value("drift_out_of_order_events_total"),
        "status": value("drift_stream_status"),
    }

window_late_events = 0
for tick in range(1, 101):
    window_late_events = module.update_stream_health(
        1,
        tick,
        rng,
        window_events_count=tick,
        window_late_events=window_late_events,
    )
warning = snapshot()

window_late_events = 0
for tick in range(101, 201):
    window_late_events = module.update_stream_health(
        2,
        tick,
        rng,
        window_events_count=tick - 100,
        window_late_events=window_late_events,
    )
critical = snapshot()

module.update_stream_health(
    0,
    201,
    rng,
    window_events_count=1,
    window_late_events=0,
)
normal = snapshot()
print(json.dumps([warning, critical, normal]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    warning, critical, normal = json.loads(completed.stdout)

    assert warning == {
        "late_total": 2.0,
        "late_rate": 0.02,
        "out_of_order_total": 1.0,
        "status": 1.0,
    }
    assert critical == {
        "late_total": 10.0,
        "late_rate": 0.08,
        "out_of_order_total": 5.0,
        "status": 2.0,
    }
    assert normal == {
        "late_total": 10.0,
        "late_rate": 0.0,
        "out_of_order_total": 5.0,
        "status": 0.0,
    }


def test_mock_max_event_gap_never_exceeds_current_window_span() -> None:
    script = r"""
import importlib.util
import json
import random
from pathlib import Path

module_path = Path("monitoring/mock_exporter/app.py")
spec = importlib.util.spec_from_file_location("mock_timing_test", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = random.Random(0)

results = []
for status in (-1, 0, 1, 2):
    current_max_gap = 0.0
    previous_span = 0.0
    previous_max_gap = 0.0
    for window_events_count in range(1, module.WINDOW_SIZE + 1):
        span, current_max_gap = module.update_window_timing(
            status,
            rng,
            window_events_count=window_events_count,
            window_max_event_gap=current_max_gap,
        )
        assert 0 <= current_max_gap <= span
        assert current_max_gap >= previous_max_gap
        assert span >= previous_span
        previous_span = span
        previous_max_gap = current_max_gap
    results.append({"status": status, "span": span, "gap": current_max_gap})

module.window_time_span_seconds.set(0)
module.max_event_gap_seconds.set(0)
results.append({
    "span": module.window_time_span_seconds._value.get(),
    "gap": module.max_event_gap_seconds._value.get(),
})
print(json.dumps(results))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    *windows, reset = json.loads(completed.stdout)

    assert all(window["gap"] <= window["span"] for window in windows)
    assert windows[-1]["gap"] >= 8.0
    assert reset == {"span": 0.0, "gap": 0.0}


def test_mock_exporter_exposes_every_dashboard_metric() -> None:
    module_path = ROOT / "monitoring/mock_exporter/app.py"
    spec = importlib.util.spec_from_file_location(
        "mock_exporter_contract_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.set_critical_scenario()

    from prometheus_client import REGISTRY

    exposition = generate_latest(REGISTRY).decode("utf-8")
    series_names = {
        line.split("{", 1)[0].split(" ", 1)[0]
        for line in exposition.splitlines()
        if line and not line.startswith("#")
    }
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
            encoding="utf-8"
        )
    )
    dashboard_metrics = _dashboard_prometheus_metrics(dashboard)
    assert dashboard_metrics - series_names == set()
