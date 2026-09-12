# Data Drift Guardian

Система контроля data drift в batch- и realtime-режимах.

## Realtime pipeline

```text
Kafka producer
    ↓
features-stream
    ↓
Kafka consumer
    ↓
WindowBuffer
    ↓
RealTimeDriftMonitor
    ↓
PrometheusExporter
    ↓
Prometheus
```

Realtime-код находится в:

```text
drift_guardian/
├── realtime/
│   ├── event.py
│   ├── producer.py
│   ├── consumer.py
│   ├── window_buffer.py
│   └── realtime_monitor.py
└── exporters/
    └── prometheus_exporter.py
```

`DriftMetricsEngine` пока не подключён: realtime-контур работает независимо и готов принимать Core analyzer через `RealTimeDriftMonitor`.

## Kafka event

```json
{
  "event_id": 123,
  "event_time": "2026-08-16T10:42:00Z",
  "age": 45,
  "income": 85000,
  "country": "DE",
  "prediction_score": 0.72
}
```

`event_time` должен содержать timezone. Дополнительные признаки должны быть плоскими JSON-значениями.

## Sliding window

По умолчанию:

```text
WINDOW_SIZE=1000
MIN_WINDOW_SIZE=300
```

Окно хранит последние `WINDOW_SIZE` валидных событий.

## Prometheus

Основные realtime-метрики:

```text
drift_events_processed_total
drift_window_size
drift_stream_status
drift_event_time_lag_seconds
drift_window_time_span_seconds
drift_max_event_gap_seconds
drift_invalid_event_time_rate
drift_late_events_total
drift_out_of_order_events_total
```

Exporter также поддерживает drift-метрики Core после его подключения.

## Установка

Требования: Python 3.13+, `uv`, Docker.

```bash
uv lock
uv sync
uv sync --group notebooks
```

## Тесты

Проверить Python-код и smoke-тесты репозитория:

```bash
uv run python -m compileall -q main.py drift_guardian tests
uv run python -m unittest discover -s tests -v
uv run python main.py
```

Проверить notebook-зависимости и чтение parquet:

```bash
uv run --group notebooks python -c "import pandas, evidently, nannyml, great_expectations, pyarrow; print('notebook dependencies: OK')"
uv run --group notebooks python -c "import pandas as pd; df = pd.read_parquet('data/train_transaction_sample.parquet'); print(df.shape)"
```

Оригинальный `notebooks/01_library_research.ipynb` перед commit лучше не сохранять после `Run All`, потому что Jupyter меняет outputs и metadata.

## End-to-end проверка

Запустить инфраструктуру:

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

Проверить Kafka topic:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --list
```

В списке должен быть `features-stream`.

Проверить exporter:

```text
http://localhost:8000/metrics
```

`drift_events_processed_total` должен расти, а `drift_window_size` — увеличиваться до `WINDOW_SIZE`.

Проверить Prometheus:

```text
http://localhost:9090/targets
```

Target `drift-consumer` должен быть `UP`.

В Prometheus можно проверить:

```promql
drift_events_processed_total
```

```promql
drift_window_size
```

```promql
increase(drift_events_processed_total[1m])
```

Последний запрос при работающем producer должен возвращать значение больше `0`.

Для быстрой проверки sliding window в PowerShell:

```powershell
docker compose down
$env:WINDOW_SIZE="20"
$env:MIN_WINDOW_SIZE="5"
$env:PRODUCER_INTERVAL_SECONDS="0.05"
docker compose up --build -d
```

После этого `drift_window_size` должен остановиться на `20`.

## Перед commit

Из существующих файлов проекта ожидаемо изменяются:

```text
.gitignore
README.md
pyproject.toml
uv.lock
```

Исходные `main.py`, `ARCHITECTURE.md`, `notebooks/` и `data/` менять не нужно.

Проверка:

```bash
git diff main --name-only -- main.py ARCHITECTURE.md notebooks data
git diff --check main
git status --short
```

Первые две команды не должны показывать изменений исходного кода.

Локальное окружение и Python-кеши исключены через `.gitignore`:

```text
.venv/
__pycache__/
*.py[cod]
```

Остановить инфраструктуру:

```bash
docker compose down -v
```