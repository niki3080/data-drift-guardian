from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from drift_guardian.batch.adversarial_validation import adversarial_validation
from drift_guardian.data_quality_checker.checker import SchemaChecker
from drift_guardian.engine import analyze_dataframe
from drift_guardian.ingestion.demo_reference import write_demo_reference

CoreAnalyzer = Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]
AdversarialAnalyzer = Callable[..., tuple[float, pd.DataFrame]]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _parse_av_thresholds(raw: Any) -> dict[str, float]:
    """Парсит ROC-AUC thresholds для AV status из runtime-конфига."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("adversarial_validation.thresholds must be a mapping")

    parsed: dict[str, float] = {}
    for level in ("warning", "critical"):
        value = raw.get(level)
        if value is not None:
            parsed[level] = float(value)

    if not parsed:
        return {}
    if set(parsed) != {"warning", "critical"}:
        raise ValueError(
            "adversarial_validation.thresholds must define warning and critical"
        )

    warning = parsed["warning"]
    critical = parsed["critical"]
    if not (0.5 <= warning < critical <= 1.0):
        raise ValueError(
            "AV ROC-AUC thresholds must satisfy "
            "0.5 <= warning < critical <= 1.0"
        )
    return parsed


def _av_status(roc_auc: float, thresholds: dict[str, float]) -> str | None:
    """Возвращает AV drift status, если thresholds явно настроены."""
    if not thresholds:
        return None
    if roc_auc >= thresholds["critical"]:
        return "critical"
    if roc_auc >= thresholds["warning"]:
        return "warning"
    return "ok"




def _importance_similarity_previous(
    current: dict[str, float],
    previous: dict[str, float],
) -> float | None:
    """Cosine similarity AV importance vectors between consecutive windows."""
    if not previous:
        return None

    features = set(current) | set(previous)
    dot = sum(current.get(name, 0.0) * previous.get(name, 0.0) for name in features)
    current_norm = math.sqrt(sum(current.get(name, 0.0) ** 2 for name in features))
    previous_norm = math.sqrt(sum(previous.get(name, 0.0) ** 2 for name in features))
    if current_norm == 0.0 and previous_norm == 0.0:
        return 1.0
    if current_norm == 0.0 or previous_norm == 0.0:
        return 0.0
    return float(dot / (current_norm * previous_norm))

def load_reference_dataframe(path: str | Path) -> pd.DataFrame:
    """Загружает reference dataset из CSV или Parquet."""
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


def load_runtime_contract(path: str | Path) -> dict[str, Any]:
    """Читает из конфига только поля, необходимые realtime-слою.

    Парсер Core здесь намеренно не используется: realtime зависит только от
    списка фич, настроек prediction и порогов, которые нужно передать дальше.
    """
    config_path = Path(path)
    if not config_path.exists():
        return {
            "monitored_columns": [],
            "thresholds": {},
            "prediction_type": None,
            "adversarial_enabled": True,
            "adversarial_thresholds": {},
        }

    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("drift config must be a YAML mapping")

    features = raw.get("features") or {}
    if not isinstance(features, dict):
        raise ValueError("config.features must be a mapping")

    monitored_columns = [str(name) for name in features]
    prediction = raw.get("prediction_metrics") or {}
    prediction_type = None
    if isinstance(prediction, dict) and prediction.get("enabled", True):
        score_column = prediction.get("score_column")
        if score_column:
            monitored_columns.append(str(score_column))
        prediction_type = prediction.get("type") or "numeric"

    thresholds = raw.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        raise ValueError("config.thresholds must be a mapping")

    normalized_thresholds: dict[str, dict[str, float]] = {}
    for metric, pair in thresholds.items():
        if not isinstance(pair, dict):
            continue
        parsed: dict[str, float] = {}
        for level in ("warning", "critical"):
            value = pair.get(level)
            if value is not None:
                parsed[level] = float(value)
        if parsed:
            normalized_thresholds[str(metric)] = parsed

    adversarial = raw.get("adversarial_validation") or {}
    if not isinstance(adversarial, dict):
        raise ValueError("config.adversarial_validation must be a mapping")
    adversarial_enabled = adversarial.get("enabled", True)
    if not isinstance(adversarial_enabled, bool):
        raise ValueError("adversarial_validation.enabled must be boolean")
    adversarial_thresholds = _parse_av_thresholds(
        adversarial.get("thresholds")
    )

    # сохраняем порядок из конфига и удаляем дубликаты
    monitored_columns = list(dict.fromkeys(monitored_columns))
    return {
        "monitored_columns": monitored_columns,
        "thresholds": normalized_thresholds,
        "prediction_type": str(prediction_type) if prediction_type else None,
        "adversarial_enabled": adversarial_enabled,
        "adversarial_thresholds": adversarial_thresholds,
    }


@dataclass(slots=True)
class EngineAdapter:
    """Связывает realtime-окно с текущим API drift-анализатора.

    Адаптер загружает reference sample, валидирует входные признаки, вызывает
    Core и дополняет отчёт runtime-полями, нужными Prometheus exporter.
    """

    reference_df: pd.DataFrame
    dataset_name: str
    profile_created_at: str
    core_analyzer: CoreAnalyzer = analyze_dataframe
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    prediction_type: str | None = None
    schema_checker: SchemaChecker | None = None
    adversarial_analyzer: AdversarialAnalyzer | None = None
    adversarial_top_features: int = 10
    adversarial_thresholds: dict[str, float] = field(default_factory=dict)
    _previous_av_importance: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        sample_size: int,
        random_state: int = 42,
        config_path: str | Path = "config/config.yaml",
        core_analyzer: CoreAnalyzer = analyze_dataframe,
        enable_schema_check: bool = True,
        enable_adversarial_validation: bool = True,
        adversarial_top_features: int = 10,
    ) -> EngineAdapter:
        """Создаёт адаптер из reference dataset и runtime-конфига."""
        if sample_size <= 0:
            raise ValueError("reference sample_size must be positive")
        if adversarial_top_features <= 0:
            raise ValueError("adversarial_top_features must be positive")

        reference_path = Path(path)
        full_reference = load_reference_dataframe(reference_path)
        if full_reference.empty:
            raise ValueError("reference dataset must not be empty")

        contract = load_runtime_contract(config_path)
        monitored_columns = contract["monitored_columns"]
        if monitored_columns:
            missing = [
                column
                for column in monitored_columns
                if column not in full_reference.columns
            ]
            if missing:
                raise ValueError(
                    "reference dataset does not match config; missing columns: "
                    + ", ".join(missing)
                )
            full_reference = full_reference[monitored_columns]

        actual_sample_size = min(sample_size, len(full_reference))
        if actual_sample_size < len(full_reference):
            reference_df = full_reference.sample(
                n=actual_sample_size,
                random_state=random_state,
            )
        else:
            reference_df = full_reference.copy()
        reference_df = reference_df.reset_index(drop=True)

        checker = None
        if enable_schema_check:
            # Фичи на уровне события считаем опциональными: отсутствующее значение
            # материализуется как None, чтобы Core мог посчитать missing_rate.
            # Для переданных значений SchemaChecker по-прежнему проверяет тип.
            checker = SchemaChecker(reference_df, required_cols=set())

        return cls(
            reference_df=reference_df,
            dataset_name=reference_path.stem,
            profile_created_at=_utc_now_iso(),
            core_analyzer=core_analyzer,
            thresholds=contract["thresholds"],
            prediction_type=contract["prediction_type"],
            schema_checker=checker,
            adversarial_analyzer=(
                adversarial_validation
                if enable_adversarial_validation and contract["adversarial_enabled"]
                else None
            ),
            adversarial_top_features=adversarial_top_features,
            adversarial_thresholds=contract["adversarial_thresholds"],
        )

    def validate_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Проверяет поля признаков события по схеме reference sample."""
        if self.schema_checker is None:
            return features
        is_valid, validated, error = self.schema_checker.check_event(features)
        if not is_valid or validated is None:
            raise ValueError(error or "event does not match reference schema")
        return validated

    def __call__(self, current_df: pd.DataFrame) -> dict[str, Any]:
        if self.schema_checker is not None:
            self.schema_checker.check_df(current_df)

        report = self.core_analyzer(self.reference_df, current_df)
        if not isinstance(report, dict):
            raise TypeError("analyze_dataframe must return dict[str, Any]")

        runtime_report = dict(report)
        runtime_report["window_size"] = len(current_df)
        if not runtime_report.get("timestamp"):
            runtime_report["timestamp"] = _utc_now_iso()

        if (
            self.adversarial_analyzer is not None
            and min(len(self.reference_df), len(current_df)) >= 3
        ):
            roc_auc, importance = self.adversarial_analyzer(
                self.reference_df,
                current_df,
            )
            top1_importance_share = float(importance.head(1)["importance"].sum())
            top3_importance_share = float(importance.head(3)["importance"].sum())
            current_importance = {
                str(row.feature): float(row.importance)
                for row in importance.itertuples(index=False)
            }
            driver_similarity_previous = _importance_similarity_previous(
                current_importance,
                self._previous_av_importance,
            )
            self._previous_av_importance = current_importance
            av_diagnostics = {
                key: importance.attrs.get(key)
                for key in (
                    "roc_auc_cv_mean",
                    "roc_auc_cv_std",
                    "roc_auc_cv_min",
                    "roc_auc_cv_max",
                    "driver_consistency",
                )
            }
            av_block: dict[str, Any] = {
                "timestamp": runtime_report["timestamp"],
                "roc_auc": roc_auc,
                "top1_importance_share": top1_importance_share,
                "top3_importance_share": top3_importance_share,
                "reference_rows": len(self.reference_df),
                "current_rows": len(current_df),
                # AV выравнивает классы, поэтому это число строк на один класс.
                "dataset_size": min(len(self.reference_df), len(current_df)),
                "features_evaluated": len(importance),
                "reference_sample_fraction": (
                    min(len(self.reference_df), len(current_df)) / len(self.reference_df)
                ),
                "current_sample_fraction": (
                    min(len(self.reference_df), len(current_df)) / len(current_df)
                ),
                "thresholds": dict(self.adversarial_thresholds),
                "feature_importance": (
                    importance.head(self.adversarial_top_features)
                    .loc[:, ["feature", "importance", "rank"]]
                    .to_dict(orient="records")
                ),
            }
            for key, value in av_diagnostics.items():
                if value is not None:
                    av_block[key] = float(value)
            if driver_similarity_previous is not None:
                av_block["driver_similarity_previous"] = driver_similarity_previous
            status = _av_status(float(roc_auc), self.adversarial_thresholds)
            if status is not None:
                av_block["status"] = status
            runtime_report["adversarial_validation"] = av_block

        report_thresholds = runtime_report.get("thresholds")
        merged_thresholds = {
            metric: dict(pair) for metric, pair in self.thresholds.items()
        }
        if isinstance(report_thresholds, dict):
            for metric, pair in report_thresholds.items():
                if isinstance(pair, dict):
                    merged_thresholds.setdefault(str(metric), {}).update(pair)
        if merged_thresholds:
            runtime_report["thresholds"] = merged_thresholds

        prediction = runtime_report.get("prediction")
        if (
            isinstance(prediction, dict)
            and self.prediction_type
            and "type" not in prediction
        ):
            runtime_report["prediction"] = {
                **prediction,
                "type": self.prediction_type,
            }
        return runtime_report

    def get_reference_metadata(self) -> dict[str, Any]:
        """Возвращает метаданные активного reference sample для Prometheus."""
        return {
            "profile_created_at": self.profile_created_at,
            "dataset_name": self.dataset_name,
            "sample_size": len(self.reference_df),
        }


def build_engine_adapter_from_env(window_size: int) -> EngineAdapter:
    """Создаёт EngineAdapter по переменным окружения realtime-сервиса."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    reference_path = Path(
        os.getenv("REFERENCE_DATA_PATH", "data/demo_reference.csv")
    )
    if not reference_path.exists() and _env_flag("GENERATE_DEMO_REFERENCE", True):
        rows = int(os.getenv("DEMO_REFERENCE_ROWS", "10000"))
        seed = int(os.getenv("DEMO_REFERENCE_SEED", "42"))
        write_demo_reference(reference_path, rows=rows, seed=seed)

    sample_multiplier = int(os.getenv("REFERENCE_SAMPLE_MULTIPLIER", "10"))
    random_state = int(os.getenv("REFERENCE_SAMPLE_RANDOM_STATE", "42"))
    config_path = os.getenv("DRIFT_CONFIG_PATH", "config/config.yaml")
    enable_schema_check = _env_flag("ENABLE_SCHEMA_CHECK", True)
    enable_adversarial_validation = _env_flag(
        "ENABLE_ADVERSARIAL_VALIDATION",
        True,
    )
    adversarial_top_features = int(os.getenv("ADVERSARIAL_TOP_FEATURES", "10"))

    if sample_multiplier <= 0:
        raise ValueError("REFERENCE_SAMPLE_MULTIPLIER must be positive")

    return EngineAdapter.from_path(
        reference_path,
        sample_size=window_size * sample_multiplier,
        random_state=random_state,
        config_path=config_path,
        enable_schema_check=enable_schema_check,
        enable_adversarial_validation=enable_adversarial_validation,
        adversarial_top_features=adversarial_top_features,
    )
