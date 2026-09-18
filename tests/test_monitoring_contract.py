from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path

import yaml
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.core.parse_config import Metric, read_config
from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.stream_metrics import load_stream_thresholds


ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(dashboard["spec"]["title"], "Data Drift Guardian v2.1")
        self.assertEqual(dashboard["apiVersion"], "dashboard.grafana.app/v2")
        self.assertEqual(dashboard["kind"], "Dashboard")
        self.assertIn(
            "Grafana v13.2.1",
            dashboard["metadata"]["annotations"]["grafana.app/saved-from-ui"],
        )

        # dashboard должен ссылаться на UID datasource из provisioning
        dashboard_text = json.dumps(dashboard, ensure_ascii=False)
        self.assertIn(str(datasource["uid"]), dashboard_text)

    def test_exporter_registers_every_metric_used_by_yulia_dashboard(self) -> None:
        dashboard_text = (
            ROOT / "monitoring/grafana/dashboards/drift_guardian.json"
        ).read_text(encoding="utf-8")
        dashboard_metrics = set(re.findall(r"\bdrift_[A-Za-z0-9_]+", dashboard_text))

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
            "drift_event_time_lag_seconds",
            "drift_events_processed_total",
            "drift_invalid_event_time_rate",
            "drift_late_events_total",
            "drift_max_event_gap_seconds",
            "drift_metric_value",
            "drift_out_of_order_events_total",
            "drift_overall_status",
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

        parsed = read_config(ROOT / "config/config.yaml")
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
        dashboard_text = (
            ROOT / "monitoring/grafana/dashboards/drift_guardian.json"
        ).read_text(encoding="utf-8")
        dashboard_metrics = set(
            re.findall(r"\bdrift_[A-Za-z0-9_]+", dashboard_text)
        )
        self.assertEqual(dashboard_metrics - series_names, set())


if __name__ == "__main__":
    unittest.main()
