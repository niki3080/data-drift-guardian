def test_broker_is_reachable(bootstrap_servers, admin_client):
    metadata = admin_client.list_topics(timeout=10)
    assert metadata.brokers, "брокеры не найдены в метаданных"


def test_topic_roundtrip(produce, topic, bootstrap_servers):
    from confluent_kafka import Consumer

    produce(topic, [{"ping": 1}])
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": "smoke",
        "auto.offset.reset": "earliest",
    })
    try:
        consumer.subscribe([topic])
        message = consumer.poll(timeout=15.0)
        assert message is not None and message.error() is None
    finally:
        consumer.close()
