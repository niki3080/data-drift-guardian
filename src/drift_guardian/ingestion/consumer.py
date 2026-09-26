from __future__ import annotations

import json
import logging
import os
import signal
import time
from threading import Event
from typing import Any, Callable

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.runtime import (
    RuntimeContext,
    analyze_current_dataframe,
    build_runtime_from_env,
)
from drift_guardian.ingestion.stream_metrics import (
    StreamTracker,
    stream_thresholds_from_config,
)
from drift_guardian.ingestion.window import WindowBuffer

LOGGER = logging.getLogger(__name__)
STOP = Event()


def stop(*_: object) -> None:
    """Запрашивает корректное завершение consumer loop."""
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
    """Ожидает доступности Kafka broker и настроенного topic."""
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
            "log_level": 0,
        }
    )
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    LOGGER.info("waiting for Kafka topic=%s bootstrap=%s", topic, bootstrap_servers)

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
        except Exception as exc:
            last_error = exc

        sleep(min(retry_interval_seconds, max(0.0, remaining)))


def _export_completed_analysis(
    exporter: PrometheusExporter,
    runtime: RuntimeContext,
    report: dict[str, Any],
    adversarial_result: tuple[float, Any] | None,
    *,
    adversarial_executed: bool,
    current_rows: int,
) -> None:
    """Экспортирует Core-report и результат планового AV-запуска."""
    exporter.update_report(report)

    if not adversarial_executed:
        # AV запускается по интервалу. На обычных окнах сохраняем последний
        # успешно опубликованный AV snapshot.
        return

    if adversarial_result is None:
        # Плановый запуск AV завершился ошибкой или не вернул результат.
        exporter.update_adversarial_validation(None)
        return

    roc_auc, importance = adversarial_result
    exporter.update_adversarial_validation_result(
        roc_auc,
        importance,
        timestamp=report.get("timestamp"),
        thresholds=runtime.adversarial_thresholds,
        reference_rows=runtime.reference_sample_size,
        current_rows=current_rows,
        top_features=runtime.adversarial_top_features,
    )


def run(runtime: RuntimeContext) -> None:
    """Читает Kafka-события и анализирует полные непересекающиеся окна."""
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
    kafka_startup_timeout = float(os.getenv("KAFKA_STARTUP_TIMEOUT_SECONDS", "60"))
    kafka_startup_retry = float(os.getenv("KAFKA_STARTUP_RETRY_SECONDS", "1"))

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
    stream_thresholds = stream_thresholds_from_config(runtime.config)
    stream = StreamTracker(
        late_event_threshold_seconds=late_threshold,
        thresholds=stream_thresholds,
    )
    exporter = PrometheusExporter()
    exporter.set_window_size(window_size)
    exporter.update_drift_thresholds(runtime.config)
    exporter.update_stream_thresholds(stream_thresholds)
    exporter.update_reference_profile(runtime.reference_metadata())
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
    LOGGER.info("consumer started topic=%s window_size=%s", topic, window_size)

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
            except InvalidEventTime as exc:
                LOGGER.warning(
                    "invalid event_time offset=%s: %s",
                    message.offset(),
                    exc,
                )
                stream.record_invalid_event_time()
                exporter.update_stream(
                    stream.snapshot(window.event_times(), ready=completed_analyses > 0)
                )
                consumer.store_offsets(message=message)
                continue
            except (
                TypeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                LOGGER.warning("invalid event offset=%s: %s", message.offset(), exc)
                consumer.store_offsets(message=message)
                continue

            window.append(event)
            exporter.set_current_window_events(len(window))
            stream.observe(event)
            exporter.record_processed_event()
            consumer.store_offsets(message=message)
            exporter.update_stream(
                stream.snapshot(window.event_times(), ready=completed_analyses > 0)
            )

            if not window.is_full:
                continue

            current_df = window.to_dataframe()
            try:
                report, adversarial_result = analyze_current_dataframe(runtime, current_df)
                _export_completed_analysis(
                    exporter,
                    runtime,
                    report,
                    adversarial_result,
                    adversarial_executed=runtime._av_executed_last_analysis,
                    current_rows=len(current_df),
                )
                exporter.record_analysis_run()
                completed_analyses += 1
            except Exception:
                LOGGER.exception(
                    "analysis failed window_size=%s — window dropped",
                    len(current_df),
                )

            # Offset подтверждается после попытки анализа: упавшее окно
            # теряется, но consumer продолжает со следующего.
            consumer.commit(asynchronous=False)
            window.clear()
            exporter.set_current_window_events(0)
            stream.reset_window()
            exporter.update_stream(
                stream.snapshot(window.event_times(), ready=True)
            )
    finally:
        consumer.close()
        metrics_server.shutdown()
        metrics_server.server_close()
        metrics_thread.join(timeout=5)
        LOGGER.info("consumer stopped")


def main() -> None:
    """Создаёт runtime-контекст и запускает realtime consumer."""
    window_size = int(os.getenv("WINDOW_SIZE", "1000"))
    runtime = build_runtime_from_env(window_size)
    run(runtime)


if __name__ == "__main__":
    main()
