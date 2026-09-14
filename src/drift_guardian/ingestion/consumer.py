from __future__ import annotations

import json
import logging
import os
import signal
import time
from threading import Event
from typing import Any, Callable

from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient

from drift_guardian.exporters.prometheus_exporter import PrometheusExporter
from drift_guardian.ingestion.engine_adapter import build_engine_adapter_from_env
from drift_guardian.ingestion.event import InvalidEventTime, KafkaEvent
from drift_guardian.ingestion.stream_metrics import (
    StreamThresholds,
    StreamTracker,
    ThresholdPair,
)
from drift_guardian.ingestion.window import Analyzer, WindowBuffer, analyze_window

LOGGER = logging.getLogger(__name__)
STOP = Event()


def stop(*_: object) -> None:
    STOP.set()


def wait_for_kafka_topic(
    bootstrap_servers: str,
    topic: str,
    *,
    timeout_seconds: float = 60.0,
    retry_interval_seconds: float = 1.0,
    admin_client_factory: Callable[[dict[str, Any]], Any] = AdminClient,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until Kafka is reachable and the configured topic exists.

    This prevents the normal local-startup race (broker socket not ready yet,
    then topic not created yet) from leaking transient librdkafka errors into
    analyzer logs. The same guard also gives a bounded, explicit failure when
    an external Kafka cluster or topic is genuinely unavailable.
    """
    if timeout_seconds <= 0:
        raise ValueError("KAFKA_STARTUP_TIMEOUT_SECONDS must be positive")
    if retry_interval_seconds <= 0:
        raise ValueError("KAFKA_STARTUP_RETRY_SECONDS must be positive")

    admin = admin_client_factory(
        {
            "bootstrap.servers": bootstrap_servers,
            # Suppress transient librdkafka stderr noise while the broker is
            # intentionally starting. We report one concise status ourselves.
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
                last_error = KafkaException(topic_metadata.error)
        except Exception as exc:  # confluent-kafka may surface several error types
            last_error = exc

        sleep(min(retry_interval_seconds, max(0.0, remaining)))


def _optional_pair(prefix: str) -> ThresholdPair | None:
    warning = os.getenv(f"{prefix}_WARNING")
    critical = os.getenv(f"{prefix}_CRITICAL")
    if warning is None and critical is None:
        return None
    if warning is None or critical is None:
        raise ValueError(
            f"Both {prefix}_WARNING and {prefix}_CRITICAL must be configured"
        )
    return ThresholdPair(warning=float(warning), critical=float(critical))


def stream_thresholds_from_env() -> StreamThresholds:
    """Temporary bridge until Lev's stream thresholds land in project config.

    No threshold values were agreed in the task discussion, so this function
    intentionally does not invent defaults.
    """
    return StreamThresholds(
        event_time_lag_seconds=_optional_pair("STREAM_EVENT_TIME_LAG_SECONDS"),
        max_event_gap_seconds=_optional_pair("STREAM_MAX_EVENT_GAP_SECONDS"),
        invalid_event_time_rate=_optional_pair("STREAM_INVALID_EVENT_TIME_RATE"),
        late_event_rate=_optional_pair("STREAM_LATE_EVENT_RATE"),
        out_of_order_event_rate=_optional_pair("STREAM_OUT_OF_ORDER_EVENT_RATE"),
    )


def _reference_metadata(analyzer: Any) -> dict[str, Any] | None:
    """Use profiler metadata when the profiler exposes the agreed hook."""
    if analyzer is None:
        return None

    provider = getattr(analyzer, "get_reference_metadata", None)
    if callable(provider):
        metadata = provider()
        return metadata if isinstance(metadata, dict) else None

    metadata = getattr(analyzer, "reference_metadata", None)
    return metadata if isinstance(metadata, dict) else None


def run(
    analyzer: Analyzer | None = None,
    reference_metadata: dict[str, Any] | None = None,
) -> None:
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
    kafka_startup_timeout = float(
        os.getenv("KAFKA_STARTUP_TIMEOUT_SECONDS", "60")
    )
    kafka_startup_retry = float(
        os.getenv("KAFKA_STARTUP_RETRY_SECONDS", "1")
    )

    if not 0 < min_window_size <= window_size:
        raise ValueError("MIN_WINDOW_SIZE must be in [1, WINDOW_SIZE]")
    if analyze_every <= 0:
        raise ValueError("ANALYZE_EVERY_N_EVENTS must be positive")

    wait_for_kafka_topic(
        bootstrap_servers,
        topic,
        timeout_seconds=kafka_startup_timeout,
        retry_interval_seconds=kafka_startup_retry,
    )

    thresholds = stream_thresholds_from_env()
    window = WindowBuffer(window_size)
    stream = StreamTracker(window_size, late_threshold, thresholds)
    exporter = PrometheusExporter()
    exporter.set_stream_thresholds(thresholds)
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
    LOGGER.info("consumer started topic=%s", topic)

    accepted_events = 0

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
                    exporter.record_analysis_run()

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
    window_size = int(os.getenv("WINDOW_SIZE", "1000"))
    analyzer = build_engine_adapter_from_env(window_size)
    run(analyzer=analyzer)


if __name__ == "__main__":
    main()
