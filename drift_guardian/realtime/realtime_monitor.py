from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from drift_guardian.realtime.event import KafkaEvent
from drift_guardian.realtime.window_buffer import WindowBuffer


class DriftEngine(Protocol):
    def analyze_dataframe(self, current_df: pd.DataFrame) -> dict[str, Any]: ...


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


class RealTimeDriftMonitor:
    def __init__(
        self,
        engine: DriftEngine | None = None,
        window_size: int = 1000,
        min_window_size: int = 300,
        late_event_threshold_seconds: float = 60.0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0 < min_window_size <= window_size:
            raise ValueError("min_window_size must be in [1, window_size]")
        if late_event_threshold_seconds < 0:
            raise ValueError("late_event_threshold_seconds must be non-negative")

        self._engine = engine
        self._window = WindowBuffer(window_size)
        self._min_window_size = min_window_size
        self._late_threshold = late_event_threshold_seconds
        self._valid_event_times = 0
        self._invalid_event_times = 0
        self._late_events = 0
        self._out_of_order_events = 0
        self._max_event_time: datetime | None = None
        self._latest_lag = 0.0

    def record_invalid_event_time(self) -> None:
        self._invalid_event_times += 1

    def update(self, event: KafkaEvent) -> dict[str, Any] | None:
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

        self._window.append(event)
        self._valid_event_times += 1

        if self._engine is None or len(self._window) < self._min_window_size:
            return None

        report = self._engine.analyze_dataframe(self._window.to_dataframe())
        report.setdefault("window_size", len(self._window))
        return report

    def stream_snapshot(self) -> StreamSnapshot:
        event_times = sorted(
            event_time.astimezone(UTC)
            for event_time in self._window.event_times()
        )
        window_time_span = 0.0
        max_event_gap = 0.0

        if len(event_times) > 1:
            window_time_span = (
                event_times[-1] - event_times[0]
            ).total_seconds()
            max_event_gap = max(
                (right - left).total_seconds()
                for left, right in zip(
                    event_times,
                    event_times[1:],
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
            status=0 if len(self._window) >= self._min_window_size else -1,
            window_size=len(self._window),
            event_time_lag_seconds=self._latest_lag,
            window_time_span_seconds=max(window_time_span, 0.0),
            max_event_gap_seconds=max(max_event_gap, 0.0),
            invalid_event_time_rate=invalid_event_time_rate,
            late_events_total=self._late_events,
            out_of_order_events_total=self._out_of_order_events,
        )
