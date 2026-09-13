from __future__ import annotations

import json
import logging
import os
import signal
from threading import Event

from confluent_kafka import Consumer, KafkaError, KafkaException

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.stream_metrics import StreamTracker
from drift_guardian.ingestion.window import Analyzer, WindowBuffer, analyze_window

LOGGER = logging.getLogger(__name__)
STOP = Event()


def stop(*_: object) -> None:
    STOP.set()


def run(analyzer: Analyzer | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.getenv("KAFKA_TOPIC", "features-stream")
    group_id = os.getenv("KAFKA_GROUP_ID", "drift-consumer")
    window_size = int(os.getenv("WINDOW_SIZE", "1000"))
    min_window_size = int(os.getenv("MIN_WINDOW_SIZE", "300"))
    analyze_every = int(os.getenv("ANALYZE_EVERY_N_EVENTS", "1"))
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
    late_threshold = float(os.getenv("LATE_EVENT_THRESHOLD_SECONDS", "60"))

    if not 0 < min_window_size <= window_size:
        raise ValueError("MIN_WINDOW_SIZE must be in [1, WINDOW_SIZE]")
    if analyze_every <= 0:
        raise ValueError("ANALYZE_EVERY_N_EVENTS must be positive")

    window = WindowBuffer(window_size)
    stream = StreamTracker(late_threshold)
    exporter = PrometheusExporter()
    metrics_server, metrics_thread = exporter.start_http_server(prometheus_port)

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

    accepted_events = 0

    try:
        while not STOP.is_set():
            message = consumer.poll(1.0)
            if message is None:
                continue

            if message.error():
                error = message.error()
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                if error.fatal():
                    raise KafkaException(error)
                LOGGER.warning("Kafka error: %s", error)
                continue

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
                stream.record_invalid_event_time()
                exporter.update_stream(
                    stream.snapshot(
                        window.event_times(),
                        len(window),
                        min_window_size,
                    )
                )
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

            window.append(event)
            stream.observe(event)
            accepted_events += 1

            exporter.update_stream(
                stream.snapshot(
                    window.event_times(),
                    len(window),
                    min_window_size,
                )
            )

            if accepted_events % analyze_every == 0:
                report = analyze_window(window, analyzer, min_window_size)
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


def main() -> None:
    run()


if __name__ == "__main__":
    main()
