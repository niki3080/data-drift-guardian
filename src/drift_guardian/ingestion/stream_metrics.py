from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from drift_guardian.ingestion.event import KafkaEvent


@dataclass(frozen=True, slots=True)
class ThresholdPair:
    warning: float
    critical: float

    def __post_init__(self) -> None:
        if self.warning < 0 or self.critical < 0:
            raise ValueError("stream thresholds must be non-negative")
        if self.warning >= self.critical:
            raise ValueError("warning threshold must be below critical threshold")


@dataclass(frozen=True, slots=True)
class StreamThresholds:
    """Optional thresholds for metrics that participate in stream status.

    Threshold values are deliberately not hard-coded here: the team discussion
    says they will be supplied by configuration, but does not define numbers.
    """

    event_time_lag_seconds: ThresholdPair | None = None
    max_event_gap_seconds: ThresholdPair | None = None
    invalid_event_time_rate: ThresholdPair | None = None
    late_event_rate: ThresholdPair | None = None
    out_of_order_event_rate: ThresholdPair | None = None

    def as_dict(self) -> dict[str, ThresholdPair]:
        pairs = {
            "event_time_lag_seconds": self.event_time_lag_seconds,
            "max_event_gap_seconds": self.max_event_gap_seconds,
            "invalid_event_time_rate": self.invalid_event_time_rate,
            "late_event_rate": self.late_event_rate,
            "out_of_order_event_rate": self.out_of_order_event_rate,
        }
        return {name: pair for name, pair in pairs.items() if pair is not None}


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    status: int
    window_size: int
    event_time_lag_seconds: float
    window_time_span_seconds: float
    max_event_gap_seconds: float
    invalid_event_time_rate: float
    late_event_rate: float
    out_of_order_event_rate: float
    late_events_total: int
    out_of_order_events_total: int


class StreamTracker:
    """Tracks stream health for the current rolling quality window.

    Window-local rates are used to derive ``drift_stream_status``. Lifetime
    counters are kept separately for Prometheus ``*_total`` series.
    """

    def __init__(
        self,
        window_size: int = 1000,
        late_event_threshold_seconds: float = 60.0,
        thresholds: StreamThresholds | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if late_event_threshold_seconds < 0:
            raise ValueError("late_event_threshold_seconds must be non-negative")

        self._late_threshold = late_event_threshold_seconds
        self.thresholds = thresholds or StreamThresholds()

        # One flag per recently observed record (valid or invalid). This keeps
        # quality rates local instead of accumulating them for process lifetime.
        self._invalid_flags: deque[int] = deque(maxlen=window_size)
        self._late_flags: deque[int] = deque(maxlen=window_size)
        self._out_of_order_flags: deque[int] = deque(maxlen=window_size)

        self._late_events_total = 0
        self._out_of_order_events_total = 0
        self._max_event_time: datetime | None = None
        self._latest_lag = 0.0

    def record_invalid_event_time(self) -> None:
        self._invalid_flags.append(1)
        self._late_flags.append(0)
        self._out_of_order_flags.append(0)

    def observe(self, event: KafkaEvent) -> None:
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

        self._invalid_flags.append(0)
        self._late_flags.append(int(is_late))
        self._out_of_order_flags.append(int(is_out_of_order))

    def snapshot(
        self,
        event_times: list[datetime],
        window_size: int,
        min_window_size: int,
    ) -> StreamSnapshot:
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

        invalid_rate = self._rate(self._invalid_flags)
        late_rate = self._rate(self._late_flags)
        out_of_order_rate = self._rate(self._out_of_order_flags)

        status = -1
        if window_size >= min_window_size:
            statuses = [0]
            values = {
                "event_time_lag_seconds": self._latest_lag,
                "max_event_gap_seconds": max_event_gap,
                "invalid_event_time_rate": invalid_rate,
                "late_event_rate": late_rate,
                "out_of_order_event_rate": out_of_order_rate,
            }
            for metric, pair in self.thresholds.as_dict().items():
                statuses.append(self._status(values[metric], pair))
            status = max(statuses)

        return StreamSnapshot(
            status=status,
            window_size=window_size,
            event_time_lag_seconds=self._latest_lag,
            window_time_span_seconds=max(window_time_span, 0.0),
            max_event_gap_seconds=max(max_event_gap, 0.0),
            invalid_event_time_rate=invalid_rate,
            late_event_rate=late_rate,
            out_of_order_event_rate=out_of_order_rate,
            late_events_total=self._late_events_total,
            out_of_order_events_total=self._out_of_order_events_total,
        )

    @staticmethod
    def _rate(flags: deque[int]) -> float:
        return sum(flags) / len(flags) if flags else 0.0

    @staticmethod
    def _status(value: float, thresholds: ThresholdPair) -> int:
        if value >= thresholds.critical:
            return 2
        if value >= thresholds.warning:
            return 1
        return 0
