import pytest

from drift_guardian.ingestion.consumer import wait_for_kafka_topic

pytestmark = pytest.mark.integration


def test_waits_until_broker_becomes_available(bootstrap_servers, topic):
    """Симулируем медленный старт внешнего брокера через фейковый admin-клиент."""
    attempts = {"count": 0}
    real_bootstrap = bootstrap_servers

    class FlakyAdmin:
        def __init__(self, config):
            from confluent_kafka.admin import AdminClient
            self._real = AdminClient({**config, "bootstrap.servers": real_bootstrap})

        def list_topics(self, topic, timeout):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("broker not ready yet")
            return self._real.list_topics(topic=topic, timeout=timeout)

    wait_for_kafka_topic(
        real_bootstrap,
        topic,
        timeout_seconds=30.0,
        retry_interval_seconds=0.2,
        admin_client_factory=FlakyAdmin,
    )
    assert attempts["count"] == 3
