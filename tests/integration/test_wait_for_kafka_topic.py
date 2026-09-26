# tests/integration/test_wait_for_kafka_topic.py
from __future__ import annotations

import pytest

from drift_guardian.ingestion.consumer import wait_for_kafka_topic

pytestmark = pytest.mark.integration


def test_returns_when_topic_exists(bootstrap_servers: str, topic: str) -> None:
    wait_for_kafka_topic(
        bootstrap_servers,
        topic,
        timeout_seconds=30.0,
        retry_interval_seconds=0.5,
    )


def test_timeout_when_topic_missing(bootstrap_servers: str) -> None:
    with pytest.raises(TimeoutError) as exc_info:
        wait_for_kafka_topic(
            bootstrap_servers,
            "topic-that-never-exists",
            timeout_seconds=3.0,
            retry_interval_seconds=0.5,
        )
    assert "Kafka startup timeout" in str(exc_info.value)


def test_timeout_when_broker_unreachable() -> None:
    # Порт заведомо закрыт — проверяем поведение при недоступном внешнем Kafka.
    with pytest.raises(TimeoutError):
        wait_for_kafka_topic(
            "127.0.0.1:1",
            "features-stream",
            timeout_seconds=3.0,
            retry_interval_seconds=0.5,
        )


@pytest.mark.parametrize(
    ("timeout", "retry"),
    [(0.0, 1.0), (-1.0, 1.0), (10.0, 0.0), (10.0, -1.0)],
)
def test_rejects_invalid_arguments(timeout: float, retry: float) -> None:
    with pytest.raises(ValueError):
        wait_for_kafka_topic("127.0.0.1:9092", "t", timeout_seconds=timeout,
                             retry_interval_seconds=retry)
