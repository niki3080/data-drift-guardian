from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from drift_guardian.core.parse_config import read_config
from drift_guardian.ingestion.event import KafkaEvent


# ------------------------------------------------------------------ #
# Контракт stream-quality
# ------------------------------------------------------------------ #
@dataclass(frozen=True, slots=True)
class ThresholdPair:
    """Пара warning/critical порогов для одной stream-метрики."""

    warning: float
    critical: float

    def __post_init__(self) -> None:
        if self.warning < 0 or self.critical < 0:
            raise ValueError("stream thresholds must be non-negative")
        if self.warning >= self.critical:
            raise ValueError("warning threshold must be below critical threshold")


@dataclass(frozen=True, slots=True)
class StreamThresholds:
    """Набор необязательных порогов качества realtime-потока."""

    event_time_lag_seconds: ThresholdPair | None = None
    window_time_span_seconds: ThresholdPair | None = None
    max_event_gap_seconds: ThresholdPair | None = None
    invalid_event_time_rate: ThresholdPair | None = None
    late_events_total: ThresholdPair | None = None
    out_of_order_events_total: ThresholdPair | None = None

    def as_dict(self) -> dict[str, ThresholdPair]:
        """Возвращает настроенные пороги с именами Prometheus-метрик."""
        pairs = {
            "drift_event_time_lag_seconds": self.event_time_lag_seconds,
            "drift_window_time_span_seconds": self.window_time_span_seconds,
            "drift_max_event_gap_seconds": self.max_event_gap_seconds,
            "drift_invalid_event_time_rate": self.invalid_event_time_rate,
            "drift_late_events_total": self.late_events_total,
            "drift_out_of_order_events_total": self.out_of_order_events_total,
        }
        return {name: pair for name, pair in pairs.items() if pair is not None}


_STREAM_CONFIG_FIELDS = {
    "drift_event_time_lag_seconds": "event_time_lag_seconds",
    "drift_window_time_span_seconds": "window_time_span_seconds",
    "drift_max_event_gap_seconds": "max_event_gap_seconds",
    "drift_invalid_event_time_rate": "invalid_event_time_rate",
    "drift_late_events_total": "late_events_total",
    "drift_out_of_order_events_total": "out_of_order_events_total",
}


def load_stream_thresholds(path: str | Path) -> StreamThresholds:
    """Читает stream_drift через общий parser Core.

    В итоговый статус входят только перечисленные в ``stream_drift`` метрики.
    Отсутствующая метрика продолжает экспортироваться, но не влияет на статус.
    """
    config_path = Path(path)
    if not config_path.exists():
        return StreamThresholds()

    config = read_config(config_path)
    stream_drift = config.stream_drift
    if stream_drift is None:
        return StreamThresholds()

    values: dict[str, ThresholdPair] = {}
    for metric_name, field_name in _STREAM_CONFIG_FIELDS.items():
        raw_pair = getattr(stream_drift, metric_name)
        if raw_pair is None:
            continue
        values[field_name] = ThresholdPair(
            warning=float(raw_pair.warning),
            critical=float(raw_pair.critical),
        )

    return StreamThresholds(**values)

@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    """Снимок метрик stream-quality для экспорта в Prometheus."""

    status: int
    event_time_lag_seconds: float
    window_time_span_seconds: float
    max_event_gap_seconds: float
    invalid_event_time_rate: float
    late_events_total: int
    out_of_order_events_total: int


# ------------------------------------------------------------------ #
# Расчёт stream-quality метрик
# ------------------------------------------------------------------ #
class StreamTracker:
    """Считает технические метрики качества текущего окна.

    Window-local invalid-rate сбрасывается после успешного анализа.
    Накопительные late/out-of-order counters сохраняются до остановки процесса.
    """

    def __init__(
        self,
        late_event_threshold_seconds: float = 60.0,
        thresholds: StreamThresholds | None = None,
    ) -> None:
        if late_event_threshold_seconds < 0:
            raise ValueError("late_event_threshold_seconds must be non-negative")

        self._late_threshold = late_event_threshold_seconds
        self.thresholds = thresholds or StreamThresholds()

        self._window_observations = 0
        self._invalid_event_times = 0

        self._late_events_total = 0
        self._out_of_order_events_total = 0
        self._max_event_time: datetime | None = None
        self._latest_lag = 0.0

    def reset_window(self) -> None:
        """Сбрасывает метрики, относящиеся только к текущему окну."""
        self._window_observations = 0
        self._invalid_event_times = 0

    def record_invalid_event_time(self) -> None:
        """Учитывает событие с невалидным event_time в текущем окне."""
        self._window_observations += 1
        self._invalid_event_times += 1

    def observe(self, event: KafkaEvent) -> None:
        """Обновляет stream-quality счётчики по валидному событию."""
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
        """Строит текущий снимок stream-quality и рассчитывает итоговый статус."""
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

        invalid_rate = self._rate(self._invalid_event_times)

        status = -1
        configured_thresholds = self.thresholds.as_dict()
        if ready and configured_thresholds:
            statuses = [0]
            values = {
                "drift_event_time_lag_seconds": self._latest_lag,
                "drift_window_time_span_seconds": max(window_time_span, 0.0),
                "drift_max_event_gap_seconds": max(max_event_gap, 0.0),
                "drift_invalid_event_time_rate": invalid_rate,
                "drift_late_events_total": float(self._late_events_total),
                "drift_out_of_order_events_total": float(
                    self._out_of_order_events_total
                ),
            }
            for metric, pair in configured_thresholds.items():
                statuses.append(self._status(values[metric], pair))
            status = max(statuses)

        return StreamSnapshot(
            status=status,
            event_time_lag_seconds=self._latest_lag,
            window_time_span_seconds=max(window_time_span, 0.0),
            max_event_gap_seconds=max(max_event_gap, 0.0),
            invalid_event_time_rate=invalid_rate,
            late_events_total=self._late_events_total,
            out_of_order_events_total=self._out_of_order_events_total,
        )

    def _rate(self, count: int) -> float:
        if self._window_observations == 0:
            return 0.0
        return count / self._window_observations

    @staticmethod
    def _status(value: float, thresholds: ThresholdPair) -> int:
        if value >= thresholds.critical:
            return 2
        if value >= thresholds.warning:
            return 1
        return 0
