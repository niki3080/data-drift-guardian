from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from drift_guardian.engine import analyze_dataframe

CoreAnalyzer = Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_reference_dataframe(path: str | Path) -> pd.DataFrame:
    """Load the reference dataset used by the realtime analyzer.

    CSV is supported for lightweight local tests; parquet is the project default.
    """
    reference_path = Path(path)
    if not reference_path.exists():
        raise FileNotFoundError(f"reference dataset not found: {reference_path}")

    suffix = reference_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(reference_path)
    if suffix == ".csv":
        return pd.read_csv(reference_path)

    raise ValueError(
        "REFERENCE_DATA_PATH must point to a .parquet or .csv file, "
        f"got: {reference_path}"
    )


@dataclass(slots=True)
class EngineAdapter:
    """Adapts Core's two-dataframe API to the realtime one-window callback."""

    reference_df: pd.DataFrame
    dataset_name: str
    profile_created_at: str
    core_analyzer: CoreAnalyzer = analyze_dataframe

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        sample_size: int,
        random_state: int = 42,
        core_analyzer: CoreAnalyzer = analyze_dataframe,
    ) -> EngineAdapter:
        if sample_size <= 0:
            raise ValueError("reference sample_size must be positive")

        reference_path = Path(path)
        reference_df = load_reference_dataframe(reference_path)
        if reference_df.empty:
            raise ValueError("reference dataset must not be empty")

        actual_sample_size = min(sample_size, len(reference_df))
        if actual_sample_size < len(reference_df):
            reference_df = reference_df.sample(
                n=actual_sample_size,
                random_state=random_state,
            )
        else:
            reference_df = reference_df.copy()

        reference_df = reference_df.reset_index(drop=True)
        return cls(
            reference_df=reference_df,
            dataset_name=reference_path.stem,
            profile_created_at=_utc_now_iso(),
            core_analyzer=core_analyzer,
        )

    def __call__(self, current_df: pd.DataFrame) -> dict[str, Any]:
        report = self.core_analyzer(self.reference_df, current_df)
        if not isinstance(report, dict):
            raise TypeError("analyze_dataframe must return dict[str, Any]")

        # Runtime-owned fields should describe the window that was actually
        # analyzed, even while Core still uses a static mock drift report.
        runtime_report = dict(report)
        runtime_report["window_size"] = len(current_df)
        runtime_report["timestamp"] = _utc_now_iso()
        return runtime_report

    def get_reference_metadata(self) -> dict[str, Any]:
        return {
            "profile_created_at": self.profile_created_at,
            "dataset_name": self.dataset_name,
            "sample_size": len(self.reference_df),
        }


def build_engine_adapter_from_env(window_size: int) -> EngineAdapter:
    """Build the default realtime/Core integration from environment settings."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    reference_path = os.getenv(
        "REFERENCE_DATA_PATH",
        "data/train_transaction_sample.parquet",
    )
    sample_multiplier = int(os.getenv("REFERENCE_SAMPLE_MULTIPLIER", "10"))
    random_state = int(os.getenv("REFERENCE_SAMPLE_RANDOM_STATE", "42"))

    if sample_multiplier <= 0:
        raise ValueError("REFERENCE_SAMPLE_MULTIPLIER must be positive")

    return EngineAdapter.from_path(
        reference_path,
        sample_size=window_size * sample_multiplier,
        random_state=random_state,
    )
