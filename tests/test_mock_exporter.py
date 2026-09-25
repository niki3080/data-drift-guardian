from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


MOCK_EXPORTER_PATH = (
    Path(__file__).resolve().parents[1] / "monitoring" / "mock_exporter" / "app.py"
)


def _load_mock_exporter() -> ModuleType:
    """Загружает mock exporter как модуль без запуска HTTP-сервера."""
    spec = importlib.util.spec_from_file_location(
        "drift_guardian_mock_exporter",
        MOCK_EXPORTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load mock exporter module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_value(
    module: ModuleType,
    *,
    feature: str,
    feature_type: str,
    metric: str,
    level: str,
) -> float:
    """Возвращает значение resolved threshold по заданному набору labels."""
    for sample in module.resolved_threshold.collect()[0].samples:
        labels = sample.labels
        if (
            labels.get("feature") == feature
            and labels.get("type") == feature_type
            and labels.get("metric") == metric
            and labels.get("level") == level
        ):
            return float(sample.value)
    raise AssertionError(
        "resolved threshold not found for "
        f"{feature}/{feature_type}/{metric}/{level}"
    )


def test_mock_exports_local_and_global_resolved_thresholds() -> None:
    """Проверяет local override и global fallback в monitoring mock."""
    module = _load_mock_exporter()
    module.set_critical_scenario()

    assert _sample_value(
        module,
        feature="age",
        feature_type="numeric",
        metric="psi",
        level="warning",
    ) == 0.05
    assert _sample_value(
        module,
        feature="age",
        feature_type="numeric",
        metric="psi",
        level="critical",
    ) == 0.12

    assert _sample_value(
        module,
        feature="income",
        feature_type="numeric",
        metric="psi",
        level="warning",
    ) == 0.1
    assert _sample_value(
        module,
        feature="income",
        feature_type="numeric",
        metric="psi",
        level="critical",
    ) == 0.25
