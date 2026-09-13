from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from drift_guardian.ingestion.event import KafkaEvent

Analyzer = Callable[[pd.DataFrame], dict[str, Any]]


class WindowBuffer:
    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self._events: deque[KafkaEvent] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._events)

    def append(self, event: KafkaEvent) -> None:
        self._events.append(event)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(event.features for event in self._events)

    def event_times(self) -> list[datetime]:
        return [event.event_time for event in self._events]


def analyze_window(
    window: WindowBuffer,
    analyzer: Analyzer | None,
    min_window_size: int,
) -> dict[str, Any] | None:
    if analyzer is None or len(window) < min_window_size:
        return None

    return analyzer(window.to_dataframe())
