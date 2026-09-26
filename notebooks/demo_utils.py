"""Воспроизводимые данные для демонстрационных notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_offline_demo_data(
    reference_rows: int = 10_000,
    current_rows: int = 2_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Создаёт reference и current с контролируемым data/prediction drift."""
    if reference_rows <= 0 or current_rows <= 0:
        raise ValueError("reference_rows and current_rows must be positive")

    rng = np.random.default_rng(seed)
    reference = pd.DataFrame(
        {
            "age": np.clip(rng.normal(40, 10, reference_rows), 18, 80).round(),
            "income": rng.lognormal(np.log(65_000), 0.35, reference_rows).round(2),
            "country": rng.choice(
                ["IT", "ES", "NL" , "PL", "FI"], 
                reference_rows, 
                p=[0.30, 0.20, 0.20, 0.20, 0.10]
            ),
            "prediction_score": rng.beta(2, 5, reference_rows),
        }
    )
    current = pd.DataFrame(
        {
            "age": np.clip(rng.normal(40, 10, current_rows), 18, 80).round(),
            "income": rng.lognormal(np.log(68_000), 0.35, current_rows).round(2),
            "country": rng.choice(
                ["IT", "ES", "NL" , "PL", "FI", "NEW_COUNTRY"],
                current_rows,
                 p=[0.29, 0.20, 0.20, 0.20, 0.10, 0.01],
            ),
            "prediction_score": rng.beta(2.15, 5, current_rows),
        }
    )

    reference_missing = rng.choice(
        reference.index, size=max(1, reference_rows // 100), replace=False
    )
    current_missing = rng.choice(
        current.index, size=round(current_rows * 0.015), replace=False
    )
    reference.loc[reference_missing, "income"] = np.nan
    current.loc[current_missing, "income"] = np.nan
    return reference, current
