"""Фикстуры интеграционных тестов realtime-consumer'а против реальной Kafka.

Брокер поднимается через testcontainers один раз на сессию и имитирует
внешний Kafka-сервис: сценарий, когда профиль local-kafka из compose не
используется, а KAFKA_BOOTSTRAP_SERVERS указывает на сторонний брокер.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("testcontainers.kafka", reason="testcontainers не установлен")

from confluent_kafka import Consumer, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

from drift_guardian.ingestion import consumer as consumer_module
from drift_guardian.ingestion.runtime import build_runtime_from_env

WINDOW_SIZE = 5
FEATURE_NAMES = ("f1", "f2", "f3")
REFERENCE_ROWS = 400


def pytest_collection_modifyitems(config, items) -> None:
    """Помечает все тесты этого пакета маркером integration."""
    for item in items:
        item.add_marker(pytest.mark.integration)


# --------------------------------------------------------------------------
# Kafka: контейнер, топики, продюсер, offset'ы
# --------------------------------------------------------------------------
import uuid

import re

from testcontainers.core.wait_strategies import LogMessageWaitStrategy
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

KAFKA_IMAGE = "apache/kafka:4.3.1"
KAFKA_PORT = 9092
KAFKA_CONTROLLER_PORT = 9093

# apache/kafka в KRaft пишет '[KafkaRaftServer nodeId=1] Kafka Server started',
# поэтому ждём свою строку вместо legacy-регулярки KafkaContainer.
KAFKA_READY_LOG = r"Kafka Server started"


class ApacheKafkaContainer(DockerContainer):
    """apache/kafka в KRaft-режиме на фиксированном порту.

    Обёртка образа при отсутствии KAFKA_ADVERTISED_LISTENERS подставляет
    туда KAFKA_LISTENERS, и 0.0.0.0 в advertised валит брокер на валидации
    ('nonroutable meta-address'). Динамический alter через kafka-configs
    невозможен: брокер не стартует без валидного advertised. Поэтому
    фиксируем маппинг портов и задаём advertised через env до старта.
    Минус: тесты упадут, если порт занят на хосте.
    """

    def __init__(self, image: str = KAFKA_IMAGE, port: int = 9092) -> None:
        super().__init__(image)
        self._port = port
        self.with_bind_ports(port, port)
        self.with_env("KAFKA_NODE_ID", "1")
        self.with_env("KAFKA_PROCESS_ROLES", "broker,controller")
        self.with_env(
            "KAFKA_CONTROLLER_QUORUM_VOTERS",
            f"1@localhost:{KAFKA_CONTROLLER_PORT}",
        )
        self.with_env("KAFKA_CONTROLLER_LISTENER_NAMES", "CONTROLLER")
        # 0.0.0.0 допустим в listeners (бинд на все интерфейсы),
        # но не в advertised — там обязателен routable-адрес.
        self.with_env(
            "KAFKA_LISTENERS",
            f"PLAINTEXT://0.0.0.0:{port},"
            f"CONTROLLER://0.0.0.0:{KAFKA_CONTROLLER_PORT}",
        )
        self.with_env(
            "KAFKA_ADVERTISED_LISTENERS", f"PLAINTEXT://localhost:{port}"
        )
        self.with_env(
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP",
            "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
        )
        self.with_env("KAFKA_INTER_BROKER_LISTENER_NAME", "PLAINTEXT")
        self.with_env("KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR", "1")
        self.with_env("KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR", "1")
        self.with_env("KAFKA_TRANSACTION_STATE_LOG_MIN_ISR", "1")
        self.with_env("KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS", "0")
        self.with_env("KAFKA_AUTO_CREATE_TOPICS_ENABLE", "false")
        self.with_env("CLUSTER_ID", uuid.uuid4().hex[:22])

    def get_bootstrap_server(self) -> str:
        # advertised=localhost:port совпадает с маппингом, поэтому
        # клиент с хоста подключается именно по этому адресу.
        return f"localhost:{self._port}"

    def start(self, timeout: float = 120.0) -> "ApacheKafkaContainer":
        """Стартует контейнер и ждёт строку готовности брокера в логах.

        Structured wait strategy вместо устаревшего wait_for_logs:
        ожидание выполняется внутри super().start(), а не отдельным вызовом.
        """
        strategy = LogMessageWaitStrategy(re.compile(KAFKA_READY_LOG))
        # Дефолтный startup timeout у стратегии — 60s; образ с холодным
        # Docker Desktop может стартовать дольше, поэтому прокидываем свой.
        strategy._startup_timeout = timeout
        self.waiting_for(strategy)
        super().start()
        return self

@pytest.fixture(autouse=True)
def reset_stop_flag() -> Iterator[None]:
    """Гарантирует чистый STOP до и после каждого теста.

    STOP — модульный threading.Event, общий на весь прогон. Без явного
    сброса тест может унаследовать установленный флаг от предыдущего:
    run() выходит из цикла на первой проверке, в логе видны только
    'consumer started' и 'consumer stopped' без обработки событий.
    """
    consumer_module.STOP.clear()
    yield
    consumer_module.STOP.set()
    consumer_module.STOP.clear()


@pytest.fixture(scope="session")
def kafka_container() -> Iterator[ApacheKafkaContainer]:
    container = ApacheKafkaContainer()
    try:
        container.start()
    except Exception:
        _dump_container_logs(container)
        container.stop()
        raise

    try:
        _wait_for_broker(container.get_bootstrap_server(), timeout=120.0)
        yield container
    finally:
        container.stop()



@pytest.fixture(scope="session")
def kafka_container() -> Iterator[ApacheKafkaContainer]:
    container = ApacheKafkaContainer()
    try:
        container.start()
    except Exception:
        _dump_container_logs(container)
        container.stop()
        raise

    try:
        _wait_for_broker(container.get_bootstrap_server(), timeout=120.0)
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def bootstrap_servers(kafka_container: ApacheKafkaContainer) -> str:
    return kafka_container.get_bootstrap_server()


def _dump_container_logs(container: DockerContainer) -> None:
    """Печатает логи контейнера: сообщение testcontainers их скрывает."""
    try:
        stdout, stderr = container.get_logs()
    except Exception as error:  # noqa: BLE001
        print(f"не удалось получить логи контейнера: {error}")
        return
    print("=== KAFKA STDOUT ===")
    print(stdout.decode("utf-8", errors="replace"))
    print("=== KAFKA STDERR ===")
    print(stderr.decode("utf-8", errors="replace"))


@pytest.fixture(scope="session")
def admin_client(bootstrap_servers: str) -> AdminClient:
    return AdminClient({"bootstrap.servers": bootstrap_servers})


@pytest.fixture
def topic(admin_client: AdminClient) -> Iterator[str]:
    """Свежий топик на каждый тест: изоляция offset'ов и состояния группы."""
    name = f"features-stream-{uuid.uuid4().hex[:8]}"
    futures = admin_client.create_topics(
        [NewTopic(name, num_partitions=1, replication_factor=1)]
    )
    futures[name].result(timeout=30)
    yield name
    admin_client.delete_topics([name])


@pytest.fixture
def group_id() -> str:
    return f"drift-consumer-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def produce(bootstrap_servers: str) -> Callable[[str, list[Any]], None]:
    """Публикует payload'ы и дожидается подтверждения брокера."""
    producer = Producer({"bootstrap.servers": bootstrap_servers})

    def _produce(topic_name: str, payloads: list[Any]) -> None:
        for payload in payloads:
            if isinstance(payload, (bytes, bytearray)):
                value = bytes(payload)
            else:
                value = json.dumps(payload).encode()
            producer.produce(topic_name, value=value)
        remaining = producer.flush(30)
        assert remaining == 0, f"producer не отправил {remaining} сообщений"

    return _produce


@pytest.fixture
def committed_offset(bootstrap_servers: str) -> Callable[[str, str], int | None]:
    """Читает закоммиченный offset группы; None — если коммита не было."""

    def _committed(group: str, topic_name: str) -> int | None:
        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group,
                "enable.auto.commit": False,
            }
        )
        try:
            partitions = consumer.committed(
                [TopicPartition(topic_name, 0)], timeout=10
            )
            offset = partitions[0].offset
            return None if offset < 0 else int(offset)
        finally:
            consumer.close()

    return _committed


def _wait_for_broker(bootstrap: str, *, timeout: float) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if admin.list_topics(timeout=2.0).brokers:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"Kafka не поднялась за {timeout}s: {last_error}")


# --------------------------------------------------------------------------
# Reference dataset и config для OfflineWrapper
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def reference_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """CSV с reference-данными: load_reference_dataframe читает .csv/.parquet."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {name: rng.normal(loc=0.0, scale=1.0, size=REFERENCE_ROWS)
         for name in FEATURE_NAMES}
    )
    path = tmp_path_factory.mktemp("reference") / "reference.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def drift_config_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Конфиг, согласованный со схемой Config.

    - thresholds заданы глобально: resolve_feature_thresholds берёт их
      как fallback для метрик фич без локальных override.
    - prediction_metrics.enabled=false: валидатор check_required_when_enabled
      не требует type/metrics/score_column.
    - stream_drift покрывает три метрики, которые проверяют тесты;
      warning < critical строго, как требует check_stream_drift_threshold_directions.
    - AV выключен: av_due() иначе вернёт True на первом окне, и LightGBM
      будет обучаться на каждом анализе.
    """
    config_text = """
thresholds:
  psi:
    warning: 0.1
    critical: 0.25

features:
  f1:
    type: numeric
    metrics: [psi]
  f2:
    type: numeric
    metrics: [psi]
  f3:
    type: numeric
    metrics: [psi]

prediction_metrics:
  enabled: false

stream_drift:
  drift_event_time_lag_seconds:
    warning: 300.0
    critical: 900.0
  drift_invalid_event_time_rate:
    warning: 0.1
    critical: 0.5
  drift_late_event_rate:
    warning: 0.2
    critical: 0.5

adversarial_validation:
  enabled: false
"""
    path = tmp_path_factory.mktemp("config") / "config.yaml"
    path.write_text(config_text.strip(), encoding="utf-8")
    return path


@pytest.fixture
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def kafka_env(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    free_port: int,
    reference_csv: Path,
    drift_config_yaml: Path,
) -> dict[str, str]:
    """Переменные окружения realtime-сервиса, которые читает consumer/runtime."""
    env = {
        "KAFKA_BOOTSTRAP_SERVERS": bootstrap_servers,
        "KAFKA_TOPIC": topic,
        "KAFKA_GROUP_ID": group_id,
        "KAFKA_STARTUP_TIMEOUT_SECONDS": "60",
        "KAFKA_STARTUP_RETRY_SECONDS": "0.5",
        "WINDOW_SIZE": str(WINDOW_SIZE),
        "LATE_EVENT_THRESHOLD_SECONDS": "60",
        "PROMETHEUS_PORT": str(free_port),
        "REFERENCE_DATA_PATH": str(reference_csv),
        "DRIFT_CONFIG_PATH": str(drift_config_yaml),
        "AV_WARNING_THRESHOLD": "0.60",
        "AV_CRITICAL_THRESHOLD": "0.75",
        "ADVERSARIAL_TOP_FEATURES": "3",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


@pytest.fixture
def runtime(kafka_env: dict[str, str]):
    """RuntimeContext из env. build_runtime_from_env принимает window_size."""
    return build_runtime_from_env(WINDOW_SIZE)


# --------------------------------------------------------------------------
# События
# --------------------------------------------------------------------------


@pytest.fixture
def make_event() -> Callable[..., dict[str, Any]]:
    """Валидное событие: event_id, timezone-aware event_time, плоские фичи.

    KafkaEvent.from_dict собирает features из всех ключей верхнего уровня,
    кроме event_id и event_time, и требует JSON-скаляры. Вложенный
    "features": {...} привёл бы к ValueError.
    """

    def _make(
        index: int = 0,
        *,
        event_time: str | dt.datetime | None = None,
        event_id: str | int | None = None,
    ) -> dict[str, Any]:
        if event_time is None:
            timestamp = dt.datetime.now(dt.UTC).isoformat()
        elif isinstance(event_time, dt.datetime):
            timestamp = event_time.isoformat()
        else:
            timestamp = event_time

        payload: dict[str, Any] = {
            "event_id": event_id if event_id is not None else f"evt-{index}",
            "event_time": timestamp,
            "f1": float(index),
            "f2": float(index % 3),
            "f3": float(index) * 0.5,
        }
        return payload

    return _make


# --------------------------------------------------------------------------
# Запуск consumer'а в отдельном потоке
# --------------------------------------------------------------------------


@pytest.fixture
def no_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """signal.signal() недоступен вне главного потока — подменяем на no-op."""
    monkeypatch.setattr(
        consumer_module.signal,
        "signal",
        lambda *_args, **_kwargs: None,
        raising=False,
    )


class ConsumerRunner:
    """Запускает consumer.run() в отдельном потоке и корректно останавливает."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> "ConsumerRunner":
        consumer_module.STOP.clear()
        self._thread = threading.Thread(
            target=self._target, name="consumer-under-test", daemon=True
        )
        self._thread.start()
        return self

    def _target(self) -> None:
        try:
            consumer_module.run(self._runtime)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc

    def stop(self, timeout: float = 30.0, *, reraise: bool = True) -> None:
        consumer_module.STOP.set()
        if self._thread is not None:
            self._thread.join(timeout)
            assert not self._thread.is_alive(), "поток consumer'а не завершился"
        if reraise and self.error is not None:
            raise self.error

    def __enter__(self) -> "ConsumerRunner":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        # Не маскируем исходное исключение из тела with.
        self.stop(reraise=exc_type is None)


@pytest.fixture
def consumer_runner() -> Iterator[Callable[[Any], ConsumerRunner]]:
    """Фабрика раннеров с гарантированной остановкой потоков после теста."""
    created: list[ConsumerRunner] = []

    def _factory(runtime: Any) -> ConsumerRunner:
        runner = ConsumerRunner(runtime)
        created.append(runner)
        return runner

    yield _factory


    # STOP выставляется один раз, чтобы разбудить все живые потоки.
    # Финальный сброс делает reset_stop_flag — здесь флаг не трогаем.
    consumer_module.STOP.set()
    for runner in created:
        runner.stop(timeout=10.0, reraise=False)

import logging


def pytest_configure(config: pytest.Config) -> None:
    """Глушит DEBUG-логи docker-клиента и testcontainers."""
    for name in ("urllib3", "docker", "testcontainers.core.docker_client"):
        logging.getLogger(name).setLevel(logging.WARNING)

@pytest.fixture
def wait_until() -> Callable[..., None]:
    def _wait(
        predicate: Callable[[], bool],
        *,
        timeout: float = 60.0,
        interval: float = 0.25,
        message: str = "условие не выполнилось",
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(interval)
        raise AssertionError(f"{message} (timeout={timeout}s)")

    return _wait

@pytest.fixture
def metric_value_strict(
    scrape_metrics: Callable[[], dict[str, float]],
) -> Callable[[str], float]:
    """Как metric_value, но падает, если /metrics недоступен.

    Защищает от чтения метрик после остановки consumer'а: run() в finally
    закрывает HTTP-сервер экспортёра, и мягкий metric_value отдаёт 0.0,
    маскируя ошибку теста под ошибку логики.
    """

    def _value(name: str) -> float:
        snapshot = scrape_metrics()
        if not snapshot:
            raise AssertionError(
                f"/metrics недоступен при чтении '{name}': "
                "consumer остановлен? читайте метрики внутри with-блока"
            )
        for candidate in (name, f"{name}_total"):
            if candidate in snapshot:
                return snapshot[candidate]
        raise AssertionError(
            f"метрика '{name}' отсутствует в /metrics; "
            f"доступны: {sorted(snapshot)[:10]}"
        )

    return _value

# --------------------------------------------------------------------------
# Скрейп Prometheus-метрик работающего consumer'а
# --------------------------------------------------------------------------


def _parse_prometheus_text(payload: str) -> dict[str, float]:
    """Парсит /metrics в {'metric{labels}': value}."""
    parsed: dict[str, float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, raw_value = line.rpartition(" ")
        if not name:
            continue
        try:
            parsed[name] = float(raw_value)
        except ValueError:
            continue
    return parsed


@pytest.fixture
def scrape_metrics(free_port: int) -> Callable[[], dict[str, float]]:
    """Снимок /metrics; пустой dict, если HTTP-сервер ещё не поднят."""

    def _scrape() -> dict[str, float]:
        url = f"http://127.0.0.1:{free_port}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return _parse_prometheus_text(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return {}

    return _scrape


@pytest.fixture
def metric_value(
    scrape_metrics: Callable[[], dict[str, float]],
) -> Callable[..., float]:
    """Значение метрики по имени; Counter'ы экспонируются с суффиксом _total."""

    def _value(name: str, *, default: float = 0.0) -> float:
        snapshot = scrape_metrics()
        for candidate in (name, f"{name}_total"):
            if candidate in snapshot:
                return snapshot[candidate]
        return default

    return _value
