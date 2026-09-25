from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drift_guardian.config_handler.parse_config import Config, read_config
from drift_guardian.ingestion.event import KafkaEvent


@dataclass(frozen=True, slots=True)
class ThresholdPair:
    """Пороговые значения warning/critical для одной stream-метрики."""

    warning: float
    critical: float

    def __post_init__(self) -> None:
        """Проверяет корректность пары warning/critical."""
        if self.warning < 0 or self.critical < 0:
            raise ValueError("stream thresholds must be non-negative")
        if self.warning >= self.critical:
            raise ValueError("warning threshold must be below critical threshold")


@dataclass(frozen=True, slots=True)
class StreamThresholds:
    """Настроенные stream-метрики, которым разрешено влиять на общий status."""

    event_time_lag_seconds: ThresholdPair | None = None
    window_time_span_seconds: ThresholdPair | None = None
    max_event_gap_seconds: ThresholdPair | None = None
    invalid_event_time_rate: ThresholdPair | None = None
    late_event_rate: ThresholdPair | None = None

    def as_dict(self) -> dict[str, ThresholdPair]:
        """Возвращает только stream-метрики с настроенными thresholds."""
        pairs = {
            "drift_event_time_lag_seconds": self.event_time_lag_seconds,
            "drift_window_time_span_seconds": self.window_time_span_seconds,
            "drift_max_event_gap_seconds": self.max_event_gap_seconds,
            "drift_invalid_event_time_rate": self.invalid_event_time_rate,
            "drift_late_event_rate": self.late_event_rate,
        }
        return {name: pair for name, pair in pairs.items() if pair is not None}


_STREAM_CONFIG_FIELDS = {
    "drift_event_time_lag_seconds": "event_time_lag_seconds",
    "drift_window_time_span_seconds": "window_time_span_seconds",
    "drift_max_event_gap_seconds": "max_event_gap_seconds",
    "drift_invalid_event_time_rate": "invalid_event_time_rate",
    "drift_late_event_rate": "late_event_rate",
}


def _pair_from_any(raw: Any) -> ThresholdPair | None:
    """Преобразует config threshold pair во внутреннюю структуру realtime."""
    if raw is None:
        return None
    warning = getattr(raw, "warning", None)
    critical = getattr(raw, "critical", None)
    if warning is None or critical is None:
        return None
    return ThresholdPair(float(warning), float(critical))


def stream_thresholds_from_config(config: Config) -> StreamThresholds:
    """Извлекает stream thresholds из уже разобранного ``Config``."""
    stream_drift = config.stream_drift
    if stream_drift is None:
        return StreamThresholds()

    values: dict[str, ThresholdPair] = {}
    for metric_name, field_name in _STREAM_CONFIG_FIELDS.items():
        pair = _pair_from_any(getattr(stream_drift, metric_name, None))
        if pair is not None:
            values[field_name] = pair
    return StreamThresholds(**values)


def load_stream_thresholds(path: str | Path) -> StreamThresholds:
    """Загружает Config через единый parser и извлекает stream thresholds."""
    config_path = Path(path)
    if not config_path.exists():
        return StreamThresholds()
    return stream_thresholds_from_config(read_config(str(config_path)))


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    """Текущий снимок технического состояния потока для Prometheus."""

    status: int
    event_time_lag_seconds: float
    window_time_span_seconds: float
    max_event_gap_seconds: float
    invalid_event_time_rate: float
    late_event_rate: float
    late_events_total: int
    out_of_order_events_total: int


class StreamTracker:
    """Накапливает stream-метрики и считает status только по настроенным порогам."""

    def __init__(
        self,
        late_event_threshold_seconds: float = 60.0,
        thresholds: StreamThresholds | None = None,
    ) -> None:
        """Инициализирует window-local и lifetime состояние stream tracker."""
        if late_event_threshold_seconds < 0:
            raise ValueError("late_event_threshold_seconds must be non-negative")

        self._late_threshold = late_event_threshold_seconds
        self.thresholds = thresholds or StreamThresholds()

        self._window_observations = 0
        self._invalid_event_times = 0
        self._late_events_window = 0

        self._late_events_total = 0
        self._out_of_order_events_total = 0
        self._max_event_time: datetime | None = None
        self._latest_lag = 0.0

    def reset_window(self) -> None:
        """Сбрасывает только window-local состояние после успешного анализа."""
        self._window_observations = 0
        self._invalid_event_times = 0
        self._late_events_window = 0

    def record_invalid_event_time(self) -> None:
        """Учитывает событие с невалидным event_time в текущем окне."""
        self._window_observations += 1
        self._invalid_event_times += 1

    def observe(self, event: KafkaEvent) -> None:
        """Обновляет window-local и lifetime stream-метрики по событию."""
        event_time = event.event_time.astimezone(UTC)
        self._latest_lag = max(
            (datetime.now(UTC) - event_time).total_seconds(),
            0.0,
        )

        is_late = self._latest_lag > self._late_threshold
        is_out_of_order = (
            self._max_event_time is not None and event_time < self._max_event_time
        )

        if is_late:
            self._late_events_window += 1
            self._late_events_total += 1
        if is_out_of_order:
            self._out_of_order_events_total += 1
        if self._max_event_time is None or event_time > self._max_event_time:
            self._max_event_time = event_time

        self._window_observations += 1

    def snapshot(
        self,
        event_times: list[datetime],
        *,
        ready: bool,
    ) -> StreamSnapshot:
        """Возвращает согласованный снимок stream-health для exporter."""
        ordered_times = sorted(event_time.astimezone(UTC) for event_time in event_times)
        window_time_span = 0.0
        max_event_gap = 0.0

        if len(ordered_times) > 1:
            window_time_span = (
                ordered_times[-1] - ordered_times[0]
            ).total_seconds()
            max_event_gap = max(
                (right - left).total_seconds()
                for left, right in zip(
                    ordered_times,
                    ordered_times[1:],
                    strict=False,
                )
            )

        valid_events = self._window_observations - self._invalid_event_times
        invalid_rate = self._rate(
            self._invalid_event_times,
            self._window_observations,
        )
        late_rate = self._rate(self._late_events_window, valid_events)

        status = -1
        configured_thresholds = self.thresholds.as_dict()
        if ready and configured_thresholds:
            values = {
                "drift_event_time_lag_seconds": self._latest_lag,
                "drift_window_time_span_seconds": max(window_time_span, 0.0),
                "drift_max_event_gap_seconds": max(max_event_gap, 0.0),
                "drift_invalid_event_time_rate": invalid_rate,
                "drift_late_event_rate": late_rate,
            }
            status = max(
                [0]
                + [
                    self._status(values[metric], pair)
                    for metric, pair in configured_thresholds.items()
                ]
            )

        return StreamSnapshot(
            status=status,
            event_time_lag_seconds=self._latest_lag,
            window_time_span_seconds=max(window_time_span, 0.0),
            max_event_gap_seconds=max(max_event_gap, 0.0),
            invalid_event_time_rate=invalid_rate,
            late_event_rate=late_rate,
            late_events_total=self._late_events_total,
            out_of_order_events_total=self._out_of_order_events_total,
        )

    @staticmethod
    def _rate(count: int, total: int) -> float:
        """Возвращает долю, безопасно обрабатывая пустой denominator."""
        return 0.0 if total == 0 else count / total

    @staticmethod
    def _status(value: float, thresholds: ThresholdPair) -> int:
        """Кодирует stream status как 0=ok, 1=warning, 2=critical."""
        if value >= thresholds.critical:
            return 2
        if value >= thresholds.warning:
            return 1
        return 0
