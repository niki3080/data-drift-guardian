import json
from pathlib import Path
from typing import Any


MOCK_REPORT_PATH = (
    Path(__file__).parents[2] / "reports" / "mock_drift_report.json"
)


def analyze_dataframe(
    reference_df: Any,
    current_df: Any,
) -> dict[str, Any]:
    """Временная заглушка до подключения настоящего drift engine.
    Создает словарь из json файла с отчетом о дрейфе."""
    with MOCK_REPORT_PATH.open(encoding="utf-8") as file:
        return json.load(file)