from collections import deque
from datetime import datetime

import pandas as pd

from drift_guardian.realtime.event import KafkaEvent


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
