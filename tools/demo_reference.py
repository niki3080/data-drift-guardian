from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def build_demo_reference(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Генерирует воспроизводимый reference dataset для локального smoke-теста."""
    if rows <= 0:
        raise ValueError("rows must be positive")

    rng = random.Random(seed)
    records = []
    for _ in range(rows):
        records.append(
            {
                "age": rng.randint(18, 75),
                "income": round(rng.uniform(30_000, 150_000), 2),
                "country": rng.choice(["DE", "FR", "EE"]),
                "prediction_score": round(rng.random(), 6),
            }
        )
    return pd.DataFrame.from_records(records)


def write_demo_reference(path: str | Path, rows: int = 10_000, seed: int = 42) -> Path:
    """Сохраняет demo reference в CSV и возвращает путь к файлу."""
    output = Path(path)
    if output.suffix.lower() != ".csv":
        raise ValueError("generated demo reference must use a .csv path")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_demo_reference(rows=rows, seed=seed).to_csv(output, index=False)
    return output


def main() -> None:
    """CLI для генерации demo reference dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo_reference.csv")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = write_demo_reference(args.output, rows=args.rows, seed=args.seed)
    print(path)


if __name__ == "__main__":
    main()
