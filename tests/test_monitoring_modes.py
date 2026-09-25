from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from prometheus_client import CollectorRegistry, generate_latest

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "docker-compose.yaml"
PROMETHEUS_PATH = ROOT / "monitoring" / "prometheus" / "prometheus.yaml"
DASHBOARD_PATH = ROOT / "monitoring" / "grafana" / "dashboards" / "drift_guardian.json"


def _dashboard() -> dict:
    """Загружает dashboard как JSON-структуру для contract-тестов."""
    return json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))


def test_prometheus_and_grafana_are_shared_services() -> None:
    """Проверяет, что базовая инфраструктура мониторинга не принадлежит profile режима."""
    services = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))["services"]

    for service_name in ("prometheus", "grafana"):
        assert "profiles" not in services[service_name]

    assert services["drift-mock-exporter"]["profiles"] == ["mock"]
    assert services["analyzer"]["profiles"] == ["realtime"]
    for service_name in ("kafka", "drift-producer"):
        assert services[service_name]["profiles"] == ["local-kafka"]



def test_prometheus_separates_mock_and_realtime_without_custom_mode_label() -> None:
    """Проверяет отдельные targets без дополнительного label в drift-метриках."""
    config = yaml.safe_load(PROMETHEUS_PATH.read_text(encoding="utf-8"))
    jobs = {item["job_name"]: item for item in config["scrape_configs"]}

    expected = {
        "drift-exporter-realtime": "analyzer:8000",
        "drift-exporter-mock": "drift-mock-exporter:8000",
    }
    for job_name, target in expected.items():
        job = jobs[job_name]
        static = job["static_configs"][0]
        assert static["targets"] == [target]
        assert "labels" not in static
        assert job["relabel_configs"] == [
            {"target_label": "job", "replacement": "drift-exporter"}
        ]


def test_history_panels_use_only_instance_active_at_dashboard_end() -> None:
    """Не смешивает историю mock и realtime и не добавляет display-label mode."""
    elements = _dashboard()["spec"]["elements"]

    for panel_id in ("panel-75", "panel-76", "panel-77", "panel-90"):
        expressions: list[str] = []

        def collect(value) -> None:
            if isinstance(value, dict):
                expr = value.get("expr")
                if isinstance(expr, str):
                    expressions.append(expr)
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)

        collect(elements[panel_id])
        panel_expr = "\n".join(expressions)
        assert 'up{job="drift-exporter"} @ end() == 1' in panel_expr
        assert "and on(instance)" in panel_expr
        assert "on(mode)" not in panel_expr


def test_dashboard_does_not_use_custom_mode_label() -> None:
    """Не допускает повторного появления технической колонки mode в Grafana."""
    dashboard_text = DASHBOARD_PATH.read_text(encoding="utf-8")
    assert "on(mode)" not in dashboard_text
    assert 'mode="mock"' not in dashboard_text
    assert 'mode="realtime"' not in dashboard_text


def test_av_driver_similarity_appears_on_second_completed_av() -> None:
    """Проверяет prev-window similarity без подстановки фиктивного первого значения."""
    registry = CollectorRegistry()
    exporter = PrometheusExporter(registry)
    thresholds = SimpleNamespace(warning=0.6, critical=0.75)

    first = pd.DataFrame(
        {
            "feature": ["age", "income"],
            "importance": [0.8, 0.2],
            "importance_std": [0.0, 0.0],
            "rank": [1, 2],
        }
    )
    first.attrs["driver_consistency"] = 0.9
    second = pd.DataFrame(
        {
            "feature": ["age", "income"],
            "importance": [0.2, 0.8],
            "importance_std": [0.0, 0.0],
            "rank": [1, 2],
        }
    )
    second.attrs["driver_consistency"] = 0.85

    exporter.update_adversarial_validation_result(
        0.55,
        first,
        timestamp="2026-09-25T01:00:00Z",
        thresholds=thresholds,
        reference_rows=1000,
        current_rows=1000,
    )
    first_metrics = generate_latest(registry).decode("utf-8")
    assert "drift_av_driver_similarity_previous -1.0" in first_metrics
    assert "drift_av_driver_consistency 0.9" in first_metrics

    exporter.update_adversarial_validation_result(
        0.56,
        second,
        timestamp="2026-09-25T01:30:00Z",
        thresholds=thresholds,
        reference_rows=1000,
        current_rows=1000,
    )
    second_metrics = generate_latest(registry).decode("utf-8")
    assert "drift_av_driver_similarity_previous -1.0" not in second_metrics
    assert "drift_av_driver_consistency 0.85" in second_metrics


def test_dashboard_uses_config_presence_for_prediction_and_names_previous_av() -> None:
    """Dashboard не путает warm-up окна с отключённым prediction monitoring."""
    dashboard = Path("monitoring/grafana/dashboards/drift_guardian.json").read_text(
        encoding="utf-8"
    )

    assert 'count(drift_threshold{metric=~\\"prediction_.+\\"})' in dashboard
    assert '"fixed": "Prev-AV sim."' in dashboard
    assert '"text": "waiting next AV"' in dashboard
