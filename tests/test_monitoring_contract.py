from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

import yaml
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.core.parse_config import Config, Metric, read_config
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


class MonitoringContractTest(unittest.TestCase):
    def test_repository_uses_yaml_extension_consistently(self) -> None:
        yml_files = [path for path in ROOT.rglob("*.yml") if ".venv" not in path.parts]
        self.assertEqual(yml_files, [])

        text_files = [
            ROOT / ".gitignore",
            ROOT / "README.md",
            ROOT / "docker-compose.yaml",
            ROOT / "monitoring/prometheus/prometheus.yaml",
        ]
        for path in text_files:
            self.assertNotIn(".yml", path.read_text(encoding="utf-8"), path)

    def test_single_compose_serves_realtime_and_mock_profiles(self) -> None:
        compose = yaml.safe_load(
            (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        )
        services = compose["services"]

        self.assertEqual(services["analyzer"]["profiles"], ["realtime"])
        self.assertEqual(services["kafka"]["profiles"], ["realtime"])
        self.assertEqual(services["kafka-init"]["profiles"], ["realtime"])
        self.assertEqual(services["drift-producer"]["profiles"], ["realtime"])
        self.assertEqual(services["drift-mock-exporter"]["profiles"], ["mock"])

        producer_dependencies = services["drift-producer"]["depends_on"]
        self.assertEqual(
            producer_dependencies["analyzer"]["condition"],
            "service_healthy",
        )
        analyzer_environment = services["analyzer"]["environment"]
        self.assertFalse(
            any(str(name).startswith("STREAM_") for name in analyzer_environment)
        )

        analyzer_dockerfile = (
            ROOT / services["analyzer"]["build"]["dockerfile"]
        ).read_text(encoding="utf-8")
        self.assertIn(
            "libgomp1",
            analyzer_dockerfile,
            "LightGBM requires the GNU OpenMP runtime in python:*-slim images",
        )

        prometheus_mounts = services["prometheus"]["volumes"]
        self.assertIn(
            "./monitoring/prometheus/prometheus.yaml:"
            "/etc/prometheus/prometheus.yaml:ro",
            prometheus_mounts,
        )

        self.assertEqual(services["prometheus"]["image"], "prom/prometheus:v3.14.0")
        self.assertEqual(services["grafana"]["image"], "grafana/grafana:13.2.1")

        grafana_mounts = services["grafana"]["volumes"]
        self.assertIn(
            "./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro",
            grafana_mounts,
        )
        self.assertIn(
            "./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro",
            grafana_mounts,
        )

    def test_single_prometheus_config_targets_shared_exporter_alias(self) -> None:
        config = yaml.safe_load(
            (ROOT / "monitoring/prometheus/prometheus.yaml").read_text(encoding="utf-8")
        )
        jobs = {job["job_name"]: job for job in config["scrape_configs"]}
        targets = jobs["drift-exporter"]["static_configs"][0]["targets"]
        self.assertEqual(targets, ["drift-exporter:8000"])

    def test_yulia_dashboard_is_provisioned_with_matching_datasource(self) -> None:
        dashboard_provider = yaml.safe_load(
            (
                ROOT
                / "monitoring/grafana/provisioning/dashboards/dashboards.yaml"
            ).read_text(encoding="utf-8")
        )
        provider = dashboard_provider["providers"][0]
        self.assertEqual(provider["options"]["path"], "/var/lib/grafana/dashboards")

        datasource = yaml.safe_load(
            (
                ROOT
                / "monitoring/grafana/provisioning/datasources/prometheus.yaml"
            ).read_text(encoding="utf-8")
        )["datasources"][0]
        self.assertEqual(datasource["url"], "http://prometheus:9090")

        dashboard = json.loads(
            (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(dashboard["spec"]["title"], "Data Drift Guardian v2.2")
        self.assertEqual(dashboard["apiVersion"], "dashboard.grafana.app/v2")
        self.assertEqual(dashboard["kind"], "Dashboard")
        self.assertIn(
            "Grafana v13.2.1",
            dashboard["metadata"]["annotations"]["grafana.app/saved-from-ui"],
        )

        # dashboard должен ссылаться на UID datasource из provisioning
        dashboard_text = json.dumps(dashboard, ensure_ascii=False)
        self.assertIn(str(datasource["uid"]), dashboard_text)

    def test_dashboard_history_defaults_and_queries_are_time_series(self) -> None:
        dashboard = json.loads(
            (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
                encoding="utf-8"
            )
        )

        time_settings = dashboard["spec"]["timeSettings"]
        self.assertEqual(time_settings["from"], "now-30m")
        self.assertEqual(time_settings["to"], "now")
        self.assertEqual(time_settings["autoRefresh"], "5s")

        for panel_id in ("panel-75", "panel-76", "panel-77", "panel-90"):
            query = (
                dashboard["spec"]["elements"][panel_id]["spec"]
                ["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]
            )
            self.assertEqual(query["format"], "time_series")
            self.assertFalse(query["instant"])
            self.assertTrue(query["range"])

        for panel_id in ("panel-76", "panel-77"):
            expression = (
                dashboard["spec"]["elements"][panel_id]["spec"]
                ["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]["expr"]
            )
            self.assertIn('feature=~"$history_feature"', expression)
            self.assertIn('metric=~"$history_metric"', expression)
            self.assertNotIn(":regex", expression)

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
        self.assertIsNotNone(history_metric)
        self.assertEqual(history_metric["allValue"], ".*")
        self.assertEqual(history_metric["query"]["spec"]["qryType"], 1)

    def test_exporter_registers_every_metric_used_by_yulia_dashboard(self) -> None:
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
            "drift_av_driver_similarity_previous",
            "drift_av_reference_rows",
            "drift_av_current_rows",
            "drift_av_dataset_size",
            "drift_av_features_evaluated",
            "drift_av_sample_fraction",
            "drift_av_threshold",
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
        self.assertEqual(dashboard_metrics, expected_metrics)
        self.assertEqual(dashboard_metrics - registered_metrics, set())

    def test_dashboard_contains_conditional_av_row_with_agreed_metrics(self) -> None:
        dashboard = json.loads(
            (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
                encoding="utf-8"
            )
        )
        rows = dashboard["spec"]["layout"]["spec"]["rows"]
        av_row = next(
            row for row in rows
            if row.get("spec", {}).get("title") == "Adversarial Validation"
        )
        conditional = av_row["spec"]["conditionalRendering"]["spec"]
        self.assertEqual(conditional["visibility"], "show")
        self.assertEqual(
            conditional["items"][0]["spec"]["variable"],
            "av_exists",
        )

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
            self.assertIn(metric, dashboard_text)

    def test_av_dashboard_matches_main_dashboard_layout(self) -> None:
        dashboard = json.loads(
            (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
                encoding="utf-8"
            )
        )
        elements = dashboard["spec"]["elements"]

        summary = elements["panel-92"]["spec"]
        self.assertEqual(summary["vizConfig"]["group"], "canvas")
        root = summary["vizConfig"]["spec"]["options"]["root"]
        self.assertEqual(root["background"]["color"]["field"], "av_status")
        self.assertFalse(summary["vizConfig"]["spec"]["options"]["inlineEditing"])
        summary_text = json.dumps(summary, ensure_ascii=False)
        for field in ("av_status", "av_status_details", "av_last_run"):
            self.assertIn(field, summary_text)
        self.assertIn("No meaningful shift", summary_text)
        self.assertNotIn("Reference and current are similar", summary_text)

        for panel_name, metric in (
            ("panel-95", "drift_av_roc_auc"),
            ("panel-98", "drift_av_roc_auc_cv_min"),
            ("panel-96", "drift_av_roc_auc_cv_std"),
            ("panel-99", "drift_av_driver_consistency"),
            ("panel-97", "drift_av_driver_similarity_previous"),
        ):
            self.assertIn(metric, json.dumps(elements[panel_name], ensure_ascii=False))
        self.assertNotIn("drift_av_dataset_size", json.dumps(elements["panel-98"], ensure_ascii=False))
        panel100_text = json.dumps(elements["panel-100"], ensure_ascii=False)
        for metric in (
            "drift_av_reference_rows",
            "drift_av_current_rows",
            "drift_av_dataset_size",
            "drift_av_features_evaluated",
            "drift_av_sample_fraction",
            "drift_av_threshold",
        ):
            self.assertIn(metric, panel100_text)
        panel97_text = json.dumps(elements["panel-97"], ensure_ascii=False)
        self.assertIn("Prev-window sim.", panel97_text)
        self.assertNotIn("Top-3 share", panel97_text)
        self.assertNotIn("CV std ↓", json.dumps(elements["panel-96"], ensure_ascii=False))
        self.assertNotIn("Driver stability ↑", json.dumps(elements["panel-99"], ensure_ascii=False))
        self.assertIn("Driver consistency", json.dumps(elements["panel-99"], ensure_ascii=False))

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
        self.assertEqual(importance_props["unit"], "percentunit")
        self.assertEqual(importance_props["decimals"], 1)
        organize = next(
            item
            for item in importance["data"]["spec"]["transformations"]
            if item.get("group") == "organize"
        )
        self.assertTrue(
            organize["spec"]["options"]["excludeByName"]["__name__"]
        )

        history = elements["panel-94"]["spec"]
        history_text = json.dumps(history, ensure_ascii=False)
        self.assertEqual(history["vizConfig"]["group"], "status-history")
        self.assertIn("drift_av_status", history_text)
        self.assertNotIn("drift_av_roc_auc", history_text)
        self.assertTrue(history["vizConfig"]["spec"]["options"]["legend"]["showLegend"])
        status_mapping = history["vizConfig"]["spec"]["fieldConfig"]["defaults"]["mappings"][0]["options"]
        self.assertEqual(status_mapping["0"]["text"], "Healthy")
        self.assertEqual(status_mapping["1"]["text"], "Warning")
        self.assertEqual(status_mapping["2"]["text"], "Critical")

        rows = dashboard["spec"]["layout"]["spec"]["rows"]
        av_row = next(
            row for row in rows
            if row.get("spec", {}).get("title") == "Adversarial Validation"
        )
        nested_rows = av_row["spec"]["layout"]["spec"]["rows"]
        latest_items = nested_rows[0]["spec"]["layout"]["spec"]["items"]
        latest_names = [item["spec"]["element"]["name"] for item in latest_items]
        self.assertEqual(
            latest_names,
            ["panel-92", "panel-95", "panel-98", "panel-96", "panel-99", "panel-97"],
        )
        self.assertEqual(len(nested_rows[1]["spec"]["layout"]["spec"]["items"]), 2)
        self.assertEqual(nested_rows[1]["spec"]["layout"]["spec"]["items"][0]["spec"]["height"], 12)
        self.assertEqual(nested_rows[2]["spec"]["title"], "AV Status Over Time")
        self.assertEqual(nested_rows[0]["spec"]["layout"]["kind"], "GridLayout")
        top_items = nested_rows[0]["spec"]["layout"]["spec"]["items"]
        self.assertEqual(len(top_items), 6)
        self.assertEqual(top_items[0]["spec"]["width"], 5)
        self.assertEqual(top_items[-2]["spec"]["width"], 4)
        self.assertEqual(top_items[-1]["spec"]["width"], 4)
        self.assertEqual(nested_rows[1]["spec"]["layout"]["spec"]["items"][0]["spec"]["width"], 12)
        self.assertEqual(nested_rows[1]["spec"]["layout"]["spec"]["items"][1]["spec"]["width"], 12)
        self.assertEqual(nested_rows[2]["spec"]["layout"]["spec"]["items"][0]["spec"]["width"], 24)

    def test_status_timelines_preserve_all_status_codes_and_av_samples(self) -> None:
        dashboard = json.loads(
            (ROOT / "monitoring/grafana/dashboards/drift_guardian.json").read_text(
                encoding="utf-8"
            )
        )
        elements = dashboard["spec"]["elements"]

        stream_history = elements["panel-91"]["spec"]
        stream_mapping = (
            stream_history["vizConfig"]["spec"]["fieldConfig"]["defaults"]
            ["mappings"][0]["options"]
        )
        self.assertEqual(stream_mapping["-1"]["text"], "Not ready")
        self.assertEqual(stream_mapping["-1"]["color"], "#ccccdb4d")
        self.assertEqual(stream_mapping["0"]["text"], "Healthy")
        self.assertEqual(stream_mapping["1"]["text"], "Degraded")
        self.assertEqual(stream_mapping["2"]["text"], "Unhealthy")
        self.assertNotIn("3", stream_mapping)

        av_history = elements["panel-94"]["spec"]
        av_query = av_history["data"]["spec"]["queries"][0]["spec"]["query"]["spec"]
        self.assertEqual(av_query["expr"], "drift_av_status")
        self.assertTrue(av_query["range"])
        self.assertFalse(av_query["instant"])
        self.assertNotIn("max_over_time", av_query["expr"])
        av_mapping = (
            av_history["vizConfig"]["spec"]["fieldConfig"]["defaults"]
            ["mappings"][0]["options"]
        )
        self.assertEqual(av_mapping["0"]["text"], "Healthy")
        self.assertEqual(av_mapping["1"]["text"], "Warning")
        self.assertEqual(av_mapping["2"]["text"], "Critical")

    def test_av_quality_kpi_labels_are_centered_over_values(self) -> None:
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
            self.assertEqual(label["config"]["align"], "center")
            self.assertEqual(value["config"]["align"], "center")
            self.assertEqual(label["placement"]["left"], value["placement"]["left"])
            self.assertEqual(label["placement"]["width"], value["placement"]["width"])

    def test_lev_threshold_contract_is_present_in_runtime_config(self) -> None:
        config = yaml.safe_load(
            (ROOT / "config/config.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["features"]["age"]["thresholds"]["psi"],
            {"warning": 0.05, "critical": 0.12},
        )
        self.assertEqual(
            config["thresholds"]["missing_rate"],
            {"warning": 0.02, "critical": 0.05},
        )
        self.assertEqual(
            config["stream_drift"],
            {
                "drift_event_time_lag_seconds": {
                    "warning": 30,
                    "critical": 120,
                },
                "drift_late_events_total": {
                    "warning": 10,
                    "critical": 50,
                },
            },
        )
        self.assertEqual(
            config["adversarial_validation"],
            {
                "enabled": True,
                "thresholds": {"warning": 0.60, "critical": 0.75},
            },
        )

        parsed = read_config(ROOT / "config/config.yaml")
        self.assertIsNotNone(parsed.adversarial_validation)
        assert parsed.adversarial_validation is not None
        self.assertEqual(parsed.adversarial_validation.thresholds.warning, 0.60)
        self.assertEqual(parsed.adversarial_validation.thresholds.critical, 0.75)
        self.assertEqual(
            parsed.features["age"].resolved_thresholds[Metric.psi].warning,
            0.05,
        )
        self.assertEqual(
            parsed.features["income"].resolved_thresholds[Metric.psi].warning,
            0.1,
        )
        self.assertEqual(
            parsed.stream_drift.drift_late_events_total.critical,
            50,
        )

        stream_thresholds = load_stream_thresholds(ROOT / "config/config.yaml")
        self.assertEqual(
            set(stream_thresholds.as_dict()),
            {"drift_event_time_lag_seconds", "drift_late_events_total"},
        )

    def test_global_threshold_is_not_overwritten_by_feature_override(self) -> None:
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
        self.assertIn(
            'drift_threshold{level="warning",metric="psi"} 0.1',
            exposition,
        )
        self.assertIn(
            'drift_threshold{level="critical",metric="psi"} 0.25',
            exposition,
        )
        self.assertNotIn(
            'drift_threshold{level="warning",metric="psi"} 0.05',
            exposition,
        )

    def test_chi2_thresholds_use_reversed_p_value_direction(self) -> None:
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
        self.assertEqual(
            parsed.features["country"].resolved_thresholds[Metric.chi2].warning,
            0.05,
        )

        invalid = {
            **valid,
            "thresholds": {
                "chi2": {"warning": 0.01, "critical": 0.05},
            },
        }
        with self.assertRaises(ValueError):
            Config(**invalid)

    def test_mock_has_many_features_mixed_statuses_and_window_variation(self) -> None:
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
        self.assertGreaterEqual(len(feature_names - {"prediction"}), 15)

        statuses = {float(value) for value in status_pattern.findall(first)}
        self.assertTrue({0.0, 1.0, 2.0}.issubset(statuses))

        common = set(first_values) & set(second_values)
        changed = sum(first_values[key] != second_values[key] for key in common)
        self.assertGreater(changed, len(common) // 2)

        self.assertIn(
            'drift_threshold{level="warning",metric="chi2"} 0.05',
            first,
        )
        self.assertIn(
            'drift_threshold{level="critical",metric="chi2"} 0.01',
            first,
        )
        self.assertIn(
            'drift_threshold{level="warning",metric="missing_rate"} 0.02',
            first,
        )
        self.assertIn(
            'drift_threshold{level="critical",metric="cardinality_ratio"} 0.6',
            first,
        )

        chi2_statuses = {
            float(value)
            for value in re.findall(
                r'^drift_status\{feature="[^"]+",metric="chi2",type="categorical"\} (-?\d+(?:\.\d+)?)$',
                first,
                re.M,
            )
        }
        self.assertTrue({1.0, 2.0}.issubset(chi2_statuses))

        self.assertIn("drift_av_available 1.0", first)
        self.assertIn("drift_av_status 0.0", first)
        self.assertIn("drift_av_status 1.0", second)
        self.assertIn("drift_av_status 2.0", third)
        self.assertIn("drift_av_roc_auc 0.54", first)
        self.assertIn("drift_av_roc_auc 0.66", second)
        self.assertIn("drift_av_roc_auc 0.8", third)
        self.assertIn("drift_av_roc_auc_cv_std 0.008", first)
        self.assertIn("drift_av_roc_auc_cv_std 0.018", second)
        self.assertIn("drift_av_roc_auc_cv_std 0.032", third)
        self.assertIn("drift_av_roc_auc_cv_min 0.529", first)
        self.assertIn("drift_av_roc_auc_cv_min 0.638", second)
        self.assertIn("drift_av_roc_auc_cv_min 0.761", third)
        self.assertIn("drift_av_driver_consistency 0.94", first)
        self.assertIn("drift_av_driver_consistency 0.86", second)
        self.assertIn("drift_av_driver_consistency 0.72", third)
        self.assertIn("drift_av_top1_importance_share 0.29", first)
        self.assertIn("drift_av_top3_importance_share 0.64", first)
        self.assertIn("drift_av_dataset_size 1000.0", first)

    def test_yulia_mock_exporter_exposes_every_dashboard_metric(self) -> None:
        module_path = ROOT / "monitoring/mock_exporter/app.py"
        spec = importlib.util.spec_from_file_location(
            "yulia_mock_exporter_contract_test",
            module_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
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
        self.assertEqual(dashboard_metrics - series_names, set())


if __name__ == "__main__":
    unittest.main()
