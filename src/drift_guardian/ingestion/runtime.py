from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from drift_guardian.analyzer.offline.offline_mode import OfflineWrapper
from drift_guardian.config_handler.parse_config import Config
# from tools.demo_reference import write_demo_reference

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AvThresholds:
    """Пороговые значения ROC AUC для monitoring-статуса AV."""

    warning: float
    critical: float

    def __post_init__(self) -> None:
        """Проверяет допустимый порядок monitoring-порогов ROC AUC."""
        if not (0.5 <= self.warning < self.critical <= 1.0):
            raise ValueError(
                "AV thresholds must satisfy 0.5 <= warning < critical <= 1.0"
            )


@dataclass(slots=True)
class RuntimeContext:
    """Контекст realtime-анализа поверх единого OfflineWrapper/Core."""

    core: OfflineWrapper
    dataset_name: str
    profile_created_at: str
    adversarial_thresholds: AvThresholds
    adversarial_top_features: int
    _last_av_monotonic: float | None = None
    _av_executed_last_analysis: bool = False

    @property
    def config(self) -> Config:
        """Возвращает единый Config, созданный Core."""
        return self.core.config

    @property
    def engine(self):
        """Возвращает экземпляр DriftMetricsEngine, созданный Core."""
        return self.core.engine

    @property
    def reference_sample_size(self) -> int:
        """Возвращает размер reference sample, доступного Core."""
        sample = self.core.reference_dict.get("sample")
        return len(sample) if isinstance(sample, pd.DataFrame) else 0

    def reference_metadata(self) -> dict[str, Any]:
        """Формирует metadata reference profile для Prometheus."""
        return {
            "dataset_name": self.dataset_name,
            "profile_created_at": self.profile_created_at,
            "sample_size": self.reference_sample_size,
        }

    def av_due(self, now_monotonic: float) -> bool:
        """Проверяет, требуется ли очередной запуск AV по настройкам Config."""
        if not self.config.adversarial_validation.enabled:
            return False

        interval_minutes = self.config.adversarial_validation.interval_minutes
        if interval_minutes is None:
            return False
        if self._last_av_monotonic is None:
            return True
        return now_monotonic - self._last_av_monotonic >= interval_minutes * 60


def _utc_now_iso() -> str:
    """Возвращает текущее UTC-время в ISO-8601 формате."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _env_flag(name: str, default: bool = False) -> bool:
    """Читает boolean-переменную окружения с явной валидацией."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def load_reference_dataframe(path: str | Path) -> pd.DataFrame:
    """Загружает reference dataset, передаваемый в Core."""
    reference_path = Path(path)
    if not reference_path.exists():
        raise FileNotFoundError(f"reference dataset not found: {reference_path}")

    suffix = reference_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(reference_path)
    if suffix == ".parquet":
        return pd.read_parquet(reference_path)
    raise ValueError(
        "REFERENCE_DATA_PATH must point to a .csv or .parquet file, "
        f"got: {reference_path}"
    )


def _monitoring_av_thresholds_from_env() -> AvThresholds:
    """Читает monitoring-пороги AV из переменных окружения."""
    return AvThresholds(
        warning=float(os.getenv("AV_WARNING_THRESHOLD", "0.60")),
        critical=float(os.getenv("AV_CRITICAL_THRESHOLD", "0.75")),
    )


def build_runtime(
    reference_df: pd.DataFrame,
    window_size: int,
    *,
    config_path: str | Path,
    dataset_name: str = "reference",
    adversarial_thresholds: AvThresholds | None = None,
    adversarial_top_features: int = 10,
) -> RuntimeContext:
    """Создаёт realtime-контекст поверх OfflineWrapper и DriftMetricsEngine.

    Включение и расписание AV определяются только блоком
    ``Config.adversarial_validation``. Realtime-слой не хранит отдельный
    переключатель AV.
    """
    if reference_df.empty:
        raise ValueError("reference dataset must not be empty")
    if adversarial_top_features <= 0:
        raise ValueError("adversarial_top_features must be positive")

    core = OfflineWrapper(
        reference_df=reference_df,
        path_to_config=str(config_path),
        take_sample=True,
        window_size=window_size
    )

    return RuntimeContext(
        core=core,
        dataset_name=dataset_name,
        profile_created_at=_utc_now_iso(),
        adversarial_thresholds=(
            adversarial_thresholds or _monitoring_av_thresholds_from_env()
        ),
        adversarial_top_features=adversarial_top_features,
    )


def build_runtime_from_env(window_size: int) -> RuntimeContext:
    """Создаёт Core-контекст из переменных окружения realtime-сервиса."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    reference_path = Path(
        os.getenv("REFERENCE_DATA_PATH", "/app/data/reference.csv")
    )
#    if not reference_path.exists() and _env_flag("GENERATE_DEMO_REFERENCE", True):
#        rows = int(os.getenv("DEMO_REFERENCE_ROWS", "10000"))
#        seed = int(os.getenv("DEMO_REFERENCE_SEED", "42"))
#        write_demo_reference(reference_path, rows=rows, seed=seed)

    # Parser конфигурации находится внутри пакета drift_guardian.
    # Пользовательский YAML монтируется в Docker по пути /app/config/config.yaml.
    config_path = Path(os.getenv("DRIFT_CONFIG_PATH", "/app/config/config.yaml"))
    reference_df = load_reference_dataframe(reference_path)

    return build_runtime(
        reference_df,
        window_size,
        config_path=config_path,
        dataset_name=reference_path.stem,
        adversarial_thresholds=_monitoring_av_thresholds_from_env(),
        adversarial_top_features=int(os.getenv("ADVERSARIAL_TOP_FEATURES", "10")),
    )


def analyze_current_dataframe(
    runtime: RuntimeContext,
    current_df: pd.DataFrame,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], tuple[float, pd.DataFrame] | None]:
    """Выполняет Core-анализ текущего окна и плановый запуск AV.

    Возвращает ``(report, av_result)``. Контекст дополнительно фиксирует факт
    реального запуска AV, чтобы exporter сохранял последний успешный AV snapshot
    на окнах, пропущенных по ``interval_minutes``.
    """
    report = runtime.core.analyze_df(current_df)
    if not isinstance(report, dict):
        raise TypeError("DriftMetricsEngine.analyze_dataframe must return dict[str, Any]")

    runtime._av_executed_last_analysis = False
    now = monotonic()
    if not runtime.av_due(now):
        return report, None

    cfg = runtime.config.adversarial_validation
    n_splits = cfg.n_splits or 3
    if len(current_df) < n_splits:
        return report, None

    runtime._last_av_monotonic = now
    runtime._av_executed_last_analysis = True
    try:
        # AV выполняется только Core и получает все параметры из единого Config.
        av_result = runtime.core.run_av(
            current_df,
            max_samples=cfg.max_samples or 100_000,
            n_splits=n_splits,
            random_state=cfg.random_state if cfg.random_state is not None else 42,
            missing_category=cfg.missing_category or "__missing__",
            lightgbm_params=cfg.lightgbm.to_params(),
        )
    except Exception:
        logger.exception(
            "adversarial validation failed; main drift report remains valid"
        )
        return report, None

    return report, av_result
