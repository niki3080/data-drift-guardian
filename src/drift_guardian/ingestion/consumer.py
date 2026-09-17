from __future__ import annotations

import json
import logging
import os
import signal
import time
from threading import Event
from typing import Any, Callable

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.engine_adapter import build_engine_adapter_from_env
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.stream_metrics import (
    StreamTracker,
    load_stream_thresholds,
)
from drift_guardian.ingestion.window import Analyzer, WindowBuffer, analyze_window

LOGGER = logging.getLogger(__name__)
STOP = Event()


# ------------------------------------------------------------------ #
# Управление процессом и готовностью Kafka
# ------------------------------------------------------------------ #
def stop(*_: object) -> None:
    """Останавливает consumer по системному сигналу."""
    STOP.set()


def wait_for_kafka_topic(
    bootstrap_servers: str,
    topic: str,
    *,
    timeout_seconds: float = 60.0,
    retry_interval_seconds: float = 1.0,
    admin_client_factory: Callable[[dict[str, Any]], Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Ждёт готовности Kafka broker и нужного топика перед запуском consumer."""
    if timeout_seconds <= 0:
        raise ValueError("KAFKA_STARTUP_TIMEOUT_SECONDS must be positive")
    if retry_interval_seconds <= 0:
        raise ValueError("KAFKA_STARTUP_RETRY_SECONDS must be positive")

    if admin_client_factory is None:
        from confluent_kafka.admin import AdminClient

        admin_client_factory = AdminClient

    admin = admin_client_factory(
        {
            "bootstrap.servers": bootstrap_servers,
            # На старте Docker broker может ещё не принимать соединения.
            # отключаем лишний шум librdkafka и оставляем собственный лог готовности
            "log_level": 0,
        }
    )
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    LOGGER.info(
        "waiting for Kafka topic=%s bootstrap=%s",
        topic,
        bootstrap_servers,
    )

    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            detail = f": {last_error}" if last_error is not None else ""
            raise TimeoutError(
                "Kafka startup timeout: "
                f"topic={topic} bootstrap={bootstrap_servers}{detail}"
            )

        try:
            metadata = admin.list_topics(
                topic=topic,
                timeout=min(2.0, max(0.1, remaining)),
            )
            topic_metadata = metadata.topics.get(topic)
            if topic_metadata is not None and topic_metadata.error is None:
                LOGGER.info(
                    "Kafka topic ready topic=%s bootstrap=%s",
                    topic,
                    bootstrap_servers,
                )
                return
            if topic_metadata is not None and topic_metadata.error is not None:
                last_error = RuntimeError(str(topic_metadata.error))
        except Exception as exc:  # confluent-kafka может вернуть разные типы ошибок
            last_error = exc

        sleep(min(retry_interval_seconds, max(0.0, remaining)))


def _reference_metadata(analyzer: Any) -> dict[str, Any] | None:
    provider = getattr(analyzer, "get_reference_metadata", None)
    if callable(provider):
        metadata = provider()
        return metadata if isinstance(metadata, dict) else None
    metadata = getattr(analyzer, "reference_metadata", None)
    return metadata if isinstance(metadata, dict) else None


def _validate_features(analyzer: Any, event: KafkaEvent) -> KafkaEvent:
    validator = getattr(analyzer, "validate_features", None)
    if not callable(validator):
        return event
    validated = validator(event.features)
    if not isinstance(validated, dict):
        raise TypeError("validate_features must return dict[str, Any]")
    return KafkaEvent(
        event_id=event.event_id,
        event_time=event.event_time,
        features=validated,
    )


# ------------------------------------------------------------------ #
# Основной consumer loop
# ------------------------------------------------------------------ #
def run(
    analyzer: Analyzer,
    reference_metadata: dict[str, Any] | None = None,
) -> None:
    """Запускает основной цикл чтения Kafka и анализа полных окон."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.getenv("KAFKA_TOPIC", "features-stream")
    group_id = os.getenv("KAFKA_GROUP_ID", "drift-consumer")
    window_size = int(os.getenv("WINDOW_SIZE", "1000"))
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
    late_threshold = float(os.getenv("LATE_EVENT_THRESHOLD_SECONDS", "60"))
    config_path = os.getenv("DRIFT_CONFIG_PATH", "config/config.yaml")
    kafka_startup_timeout = float(
        os.getenv("KAFKA_STARTUP_TIMEOUT_SECONDS", "60")
    )
    kafka_startup_retry = float(
        os.getenv("KAFKA_STARTUP_RETRY_SECONDS", "1")
    )

    if window_size <= 0:
        raise ValueError("WINDOW_SIZE must be positive")

    from confluent_kafka import Consumer, KafkaError, KafkaException

    wait_for_kafka_topic(
        bootstrap_servers,
        topic,
        timeout_seconds=kafka_startup_timeout,
        retry_interval_seconds=kafka_startup_retry,
    )

    window = WindowBuffer(window_size)
    stream_thresholds = load_stream_thresholds(config_path)
    stream = StreamTracker(
        late_event_threshold_seconds=late_threshold,
        thresholds=stream_thresholds,
    )
    exporter = PrometheusExporter()
    exporter.set_window_size(window_size)
    exporter.update_stream_thresholds(stream_thresholds)
    exporter.update_reference_profile(
        reference_metadata or _reference_metadata(analyzer)
    )
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
    LOGGER.info(
        "consumer started topic=%s window_size=%s",
        topic,
        window_size,
    )

    completed_analyses = 0

    try:
        while not STOP.is_set():
            message = consumer.poll(1.0)
            if message is None:
                exporter.refresh_analysis_age()
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
                event = _validate_features(analyzer, event)
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
                        ready=completed_analyses > 0,
                    )
                )
                consumer.store_offsets(message=message)
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
                consumer.store_offsets(message=message)
                continue

            window.append(event)
            stream.observe(event)
            exporter.record_processed_event()
            consumer.store_offsets(message=message)

            exporter.update_stream(
                stream.snapshot(
                    window.event_times(),
                    ready=completed_analyses > 0,
                )
            )

            if window.is_full:
                report = analyze_window(window, analyzer)
                if report is None:
                    raise RuntimeError("full analysis window produced no report")

                # анализ считаем завершённым только после успешного Core
                # и экспорта отчёта.
                exporter.update_report(report)
                exporter.record_analysis_run()
                completed_analyses += 1

                # подтверждаем offsets только после полного успешного окна,
                # затем начинаем собирать следующее непересекающееся окно.
                consumer.commit(asynchronous=False)
                window.clear()
                stream.reset_window()
    finally:
        consumer.close()
        metrics_server.shutdown()
        metrics_server.server_close()
        metrics_thread.join(timeout=5)
        LOGGER.info("consumer stopped")


def main() -> None:
    """Создаёт analyzer из env и запускает realtime consumer."""
    window_size = int(os.getenv("WINDOW_SIZE", "1000"))
    analyzer = build_engine_adapter_from_env(window_size)
    run(analyzer=analyzer)


if __name__ == "__main__":
    main()
