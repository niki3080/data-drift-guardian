import argparse
import json
import os
import random
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pandas as pd
import yaml
from confluent_kafka import Producer

DEFAULT_DRIFT_CONFIG_PATH = Path(__file__).parent / "drift_config.yaml"
VALID_DRIFT_TYPES = {"shift", "scale", "noise", "categorical_swap"}

STOP = Event()


def stop(*_: object) -> None:
    """Останавливает demo producer по системному сигналу."""
    STOP.set()


def load_reference_dataset(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Reference dataset not found at {path}")

    df = pd.read_parquet(path_obj) if path_obj.suffix == ".parquet" else pd.read_csv(path_obj)
    print(f"Loaded reference dataset: {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_drift_config(path: str | None) -> dict:
    config_path = Path(path) if path else DEFAULT_DRIFT_CONFIG_PATH
    if not config_path.exists():
        print(f"Drift config not found at {config_path}, drift simulation disabled")
        return {}

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    features = raw.get("features", {})
    _validate_drift_config(features)
    print(f"Loaded drift config for features: {list(features.keys())}" if features
          else "Drift config is empty, drift simulation disabled")
    return features


def _validate_drift_config(features: dict) -> None:
    for name, cfg in features.items():
        drift_type = cfg.get("type")
        if drift_type not in VALID_DRIFT_TYPES:
            raise ValueError(f"Unknown drift type '{drift_type}' for feature '{name}'")
        if drift_type != "categorical_swap" and "magnitude" not in cfg:
            raise ValueError(f"Feature '{name}': 'magnitude' is required for type '{drift_type}'")
        if drift_type == "categorical_swap":
            for key in ("probability", "target_category"):
                if key not in cfg:
                    raise ValueError(f"Feature '{name}': '{key}' is required for categorical_swap")


def apply_drift(row: dict, drift_config: dict, step: int) -> dict:
    if not drift_config:
        return row

    drifted = dict(row)
    for feature, cfg in drift_config.items():
        if feature not in drifted:
            continue

        start_step = cfg.get("start_step", 0)
        if step < start_step:
            continue

        ramp_steps = max(cfg.get("ramp_steps", 1), 1)
        progress = min(1.0, (step - start_step) / ramp_steps)
        drift_type = cfg["type"]

        if drift_type == "shift":
            drifted[feature] = drifted[feature] + cfg["magnitude"] * progress
        elif drift_type == "scale":
            drifted[feature] = drifted[feature] * (1 + cfg["magnitude"] * progress)
        elif drift_type == "noise":
            drifted[feature] = drifted[feature] + random.gauss(0, cfg["magnitude"] * progress)
        elif drift_type == "categorical_swap":
            if random.random() < cfg["probability"] * progress:
                drifted[feature] = cfg["target_category"]

    return drifted


def sanitize_types(row: dict) -> dict:
    """Приводит numpy/pandas типы к чистым Python-типам перед сериализацией в JSON."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def build_event(event_id: int, reference_df: pd.DataFrame | None, drift_config: dict, step: int) -> dict[str, object]:
    """Генерирует одно demo-событие: из референса (если задан) или синтетически,
    затем применяет дрифт и приводит типы."""
    if reference_df is not None:
        row = reference_df.sample(1).iloc[0].to_dict()
    else:
        row = {
            "age": random.randint(18, 75),
            "income": round(random.uniform(30_000, 150_000), 2),
            "country": random.choice(["DE", "FR", "EE"]),
            "prediction_score": round(random.random(), 6),
        }

    row = apply_drift(row, drift_config, step)
    row = sanitize_types(row)

    row["event_id"] = event_id
    row["event_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drift Guardian demo producer")
    parser.add_argument("--reference-path", default=os.environ.get("REFERENCE_DATASET_PATH"))
    parser.add_argument("--drift-config-path", default=os.environ.get("DRIFT_CONFIG_PATH"))
    parser.add_argument("--interval", type=float,
                         default=float(os.environ.get("PRODUCER_INTERVAL_SECONDS", "0.2")))
    parser.add_argument("--seed", type=int,
                         default=int(os.environ["RANDOM_SEED"]) if os.environ.get("RANDOM_SEED") else 42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.getenv("KAFKA_TOPIC", "features-stream")

    reference_df = load_reference_dataset(args.reference_path) if args.reference_path else None
    drift_config = load_drift_config(args.drift_config_path)

    producer = Producer({"bootstrap.servers": bootstrap_servers})

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    event_id = 1
    step = 0
    active_drift_features: set[str] = set()

    try:
        while not STOP.is_set():
            event = build_event(event_id, reference_df, drift_config, step)

            for feature, cfg in drift_config.items():
                if step == cfg.get("start_step", 0) and feature not in active_drift_features:
                    active_drift_features.add(feature)
                    print(f"[step {step}] Drift started for '{feature}' (type={cfg['type']})")

            producer.produce(
                topic,
                key=str(event_id),
                value=json.dumps(event).encode(),
            )
            producer.poll(0)

            event_id += 1
            step += 1
            STOP.wait(args.interval)
    finally:
        producer.flush(10)


if __name__ == "__main__":
    main()
