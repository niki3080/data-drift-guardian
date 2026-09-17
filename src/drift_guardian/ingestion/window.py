from __future__ import annotations

from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from drift_guardian.ingestion.event import KafkaEvent

Analyzer = Callable[[pd.DataFrame], dict[str, Any]]


class WindowBuffer:
    """Непересекающееся count-based окно фиксированного размера."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._events: deque[KafkaEvent] = deque()

    def __len__(self) -> int:
        return len(self._events)

    @property
    def is_full(self) -> bool:
        """Показывает, готово ли окно к анализу."""
        return len(self._events) == self.max_size

    def append(self, event: KafkaEvent) -> None:
        """Добавляет событие в незаполненное окно."""
        if self.is_full:
            raise RuntimeError("analysis window is full and must be analyzed/reset")
        self._events.append(event)

    def clear(self) -> None:
        """Очищает окно после успешного анализа."""
        self._events.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Преобразует feature-поля окна в pandas DataFrame."""
        return pd.DataFrame(event.features for event in self._events)

    def event_times(self) -> list[datetime]:
        """Возвращает event_time всех событий текущего окна."""
        return [event.event_time for event in self._events]


def analyze_window(
    window: WindowBuffer,
    analyzer: Analyzer | None,
) -> dict[str, Any] | None:
    """Запускает analyzer только для полностью собранного окна."""
    if analyzer is None or not window.is_full:
        return None
    return analyzer(window.to_dataframe())
