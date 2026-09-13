from dataclasses import dataclass
from datetime import UTC, datetime

from drift_guardian.ingestion.event import KafkaEvent


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    status: int
    window_size: int
    event_time_lag_seconds: float
    window_time_span_seconds: float
    max_event_gap_seconds: float
    invalid_event_time_rate: float
    late_events_total: int
    out_of_order_events_total: int


class StreamTracker:
    def __init__(self, late_event_threshold_seconds: float = 60.0) -> None:
        if late_event_threshold_seconds < 0:
            raise ValueError("late_event_threshold_seconds must be non-negative")

        self._late_threshold = late_event_threshold_seconds
        self._valid_event_times = 0
        self._invalid_event_times = 0
        self._late_events = 0
        self._out_of_order_events = 0
        self._max_event_time: datetime | None = None
        self._latest_lag = 0.0

    def record_invalid_event_time(self) -> None:
        self._invalid_event_times += 1

    def observe(self, event: KafkaEvent) -> None:
        event_time = event.event_time.astimezone(UTC)
        self._latest_lag = max(
            (datetime.now(UTC) - event_time).total_seconds(),
            0.0,
        )

        if self._latest_lag > self._late_threshold:
            self._late_events += 1

        if self._max_event_time is not None and event_time < self._max_event_time:
            self._out_of_order_events += 1

        if self._max_event_time is None or event_time > self._max_event_time:
            self._max_event_time = event_time

        self._valid_event_times += 1

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

        total_event_times = self._valid_event_times + self._invalid_event_times
        invalid_event_time_rate = (
            self._invalid_event_times / total_event_times
            if total_event_times
            else 0.0
        )

        return StreamSnapshot(
            status=0 if window_size >= min_window_size else -1,
            window_size=window_size,
            event_time_lag_seconds=self._latest_lag,
            window_time_span_seconds=max(window_time_span, 0.0),
            max_event_gap_seconds=max(max_event_gap, 0.0),
            invalid_event_time_rate=invalid_event_time_rate,
            late_events_total=self._late_events,
            out_of_order_events_total=self._out_of_order_events,
        )
