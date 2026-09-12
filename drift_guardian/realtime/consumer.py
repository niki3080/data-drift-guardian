import json
import logging
import os
import signal
from threading import Event

from confluent_kafka import Consumer, KafkaError, KafkaException

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.realtime.event import InvalidEventTime, KafkaEvent
from drift_guardian.realtime.realtime_monitor import RealTimeDriftMonitor

LOGGER = logging.getLogger(__name__)
STOP = Event()


def stop(*_: object) -> None:
    STOP.set()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.getenv("KAFKA_TOPIC", "features-stream")
    group_id = os.getenv("KAFKA_GROUP_ID", "drift-consumer")
    window_size = int(os.getenv("WINDOW_SIZE", "1000"))
    min_window_size = int(os.getenv("MIN_WINDOW_SIZE", "300"))
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
    late_threshold = float(
        os.getenv("LATE_EVENT_THRESHOLD_SECONDS", "60")
    )

    monitor = RealTimeDriftMonitor(
        engine=None,
        window_size=window_size,
        min_window_size=min_window_size,
        late_event_threshold_seconds=late_threshold,
    )
    exporter = PrometheusExporter()
    metrics_server, metrics_thread = exporter.start_http_server(
        prometheus_port
    )

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
        }
    )
    consumer.subscribe([topic])

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOGGER.info("consumer started topic=%s", topic)

    try:
        while not STOP.is_set():
            message = consumer.poll(1.0)
            if message is None:
                continue

            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(message.error())

            try:
                payload = json.loads(message.value())
                if not isinstance(payload, dict):
                    raise ValueError("Kafka message must contain a JSON object")
                event = KafkaEvent.from_dict(payload)
            except InvalidEventTime as exc:
                LOGGER.warning(
                    "invalid event_time offset=%s: %s",
                    message.offset(),
                    exc,
                )
                monitor.record_invalid_event_time()
                exporter.update_stream(monitor.stream_snapshot())
                consumer.commit(message=message, asynchronous=False)
                continue
            except (
                TypeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                LOGGER.warning(
                    "invalid event offset=%s: %s",
                    message.offset(),
                    exc,
                )
                consumer.commit(message=message, asynchronous=False)
                continue

            report = monitor.update(event)
            exporter.update_stream(monitor.stream_snapshot())
            if report is not None:
                exporter.update_report(report)

            consumer.store_offsets(message=message)
            consumer.commit(asynchronous=False)
            exporter.record_processed_event()
    finally:
        consumer.close()
        metrics_server.shutdown()
        metrics_server.server_close()
        metrics_thread.join(timeout=5)
        LOGGER.info("consumer stopped")


if __name__ == "__main__":
    main()
