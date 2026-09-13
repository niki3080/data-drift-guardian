# Data Drift Guardian

Система мониторинга data drift для batch- и realtime-сценариев.

## Архитектура

Realtime-контур:

```text
Kafka producer
    ↓
features-stream
    ↓
Kafka consumer
    ↓
sliding window
    ↓
pandas.DataFrame
    ↓
DriftMetricsEngine.analyze_dataframe(current_df)
    ↓
drift_report
    ↓
PrometheusExporter
    ↓
Prometheus
    ↓
Grafana
```

Зона realtime-части:

- приём событий из Kafka;
- валидация входного события;
- count-based sliding window;
- сбор окна в `pandas.DataFrame` до analyzer;
- stream/time-метрики;
- экспорт `drift_report` в Prometheus;
- Docker-сборка сервиса и локальный Kafka producer для smoke-тестов.

Расчёт drift-метрик находится в analyzer/Core. Grafana и визуализация находятся
в monitoring-блоке.

## Структура realtime-блока

```text
src/drift_guardian/
├── ingestion/
│   ├── consumer.py
│   ├── event.py
│   ├── producer.py
│   ├── stream_metrics.py
│   └── window.py
├── exporters/
│   └── prometheus_exporter.py
└── Dockerfile

monitoring/
└── prometheus/
    └── prometheus.yaml

tests/
└── test_ingestion.py

docker-compose.yaml
```

Зависимости проекта хранятся в общем `pyproject.toml`. Отдельный
`requirements.txt` для ingestion не используется.

## Kafka event

Пример события:

```json
{
  "event_id": 123,
  "event_time": "2026-09-13T11:32:00Z",
  "age": 45,
  "income": 85000,
  "country": "DE",
  "prediction_score": 0.72
}
```

Требования:

- `event_id` — строка или целое число;
- `event_time` — timezone-aware ISO-8601 timestamp;
- остальные поля — плоские JSON scalar-значения;
- вложенные `list` и `dict` отклоняются;
- `prediction_score`, если передан, должен быть числом в диапазоне `[0, 1]`.

`event_id` и `event_time` не попадают в DataFrame для analyzer.
`prediction_score` остаётся колонкой DataFrame для prediction drift.

## Sliding window

Параметры по умолчанию:

```text
WINDOW_SIZE=1000
MIN_WINDOW_SIZE=300
ANALYZE_EVERY_N_EVENTS=1
```

Окно хранит последние `WINDOW_SIZE` валидных событий. После заполнения новое
событие вытесняет самое старое.

Analyzer вызывается только после достижения `MIN_WINDOW_SIZE`:

```python
report = analyzer(window.to_dataframe())
```

Интерфейс соответствует новому контракту Core:

```python
def analyze_dataframe(current_df: pandas.DataFrame) -> dict:
    ...
```

Пока новый `DriftMetricsEngine` не подключён в `main`, consumer запускается без
analyzer. Kafka, sliding window и stream-метрики при этом работают независимо.

## Drift report

Exporter ожидает report следующего типа:

```json
{
  "timestamp": "2026-09-13T11:32:00Z",
  "window_size": 1000,
  "overall_status": "critical",
  "active_alerts": 3,
  "features": {
    "age": {
      "type": "numeric",
      "status": "critical",
      "metrics": {
        "psi": {
          "value": 0.31,
          "warning": 0.1,
          "critical": 0.25,
          "status": "critical"
        }
      }
    }
  },
  "prediction": {
    "status": "critical",
    "metrics": {
      "prediction_psi": {
        "value": 0.29,
        "warning": 0.1,
        "critical": 0.25,
        "status": "critical"
      }
    }
  }
}
```

Набор конкретных drift-метрик не зашит в exporter: он экспортирует любую
метрику из `features[*].metrics` и `prediction.metrics`, если она соответствует
этому контракту.

## Prometheus metrics

Drift report:

```text
drift_overall_status
drift_active_alerts
drift_report_timestamp_seconds
drift_metric_value{feature,metric}
drift_threshold{feature,metric,level}
drift_status{feature,metric}
drift_status_feature{feature}
```

Статусы:

```text
-1 = insufficient_data
 0 = ok
 1 = warning
 2 = critical
```

Stream metrics:

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

## Установка

Требования:

- Python 3.13+;
- `uv`;
- Docker Desktop или Docker Engine;
- Docker Compose v2.

Создать окружение и установить зависимости:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv lock
uv sync
```

Для notebook-зависимостей:

```powershell
uv sync --group notebooks
```

## Python tests

```powershell
uv run python -m compileall -q src tests
uv run python -m unittest discover -s tests -v
```

Тесты ingestion проверяют:

- Kafka event contract;
- timezone-aware `event_time`;
- sliding window и вытеснение старых событий;
- преобразование окна в DataFrame до analyzer;
- stream/time-метрики;
- новый `drift_report` → Prometheus contract.

Проверка существующего entrypoint проекта:

```powershell
uv run python main.py
```

Проверка offline report:

```powershell
uv run drift-guardian-report reports/mock_drift_report.json "$env:TEMP\drift_guardian_test_report.html"
Remove-Item "$env:TEMP\drift_guardian_test_report.html"
```

Проверка notebook-зависимостей и sample dataset:

```powershell
uv run --group notebooks python -c "import pandas, evidently, nannyml, great_expectations, pyarrow; print('notebook dependencies: OK')"
uv run --group notebooks python -c "import pandas as pd; df = pd.read_parquet('data/train_transaction_sample.parquet'); print(df.shape); print(df.columns.tolist())"
```

Не запускайте `Run All` на оригинальном notebook перед commit: Jupyter может
изменить `execution_count`, outputs и metadata. Для полного запуска используйте
временную копию notebook и данных.

## Docker smoke test

Проверить compose:

```powershell
docker compose -f docker-compose.yaml config
```

Поднять общий stack с локальной Kafka и demo producer:

```powershell
docker compose -f docker-compose.yaml --profile kafka up --build -d
docker compose -f docker-compose.yaml ps
```

Проверить Kafka topic:

```powershell
docker compose -f docker-compose.yaml exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --list
```

В списке должен быть:

```text
features-stream
```

Проверить логи:

```powershell
docker compose -f docker-compose.yaml logs --tail=50 drift-producer
docker compose -f docker-compose.yaml logs --tail=50 analyzer
```

## Проверка exporter

Открыть:

```text
http://localhost:8000/metrics
```

При работающем producer:

- `drift_events_processed_total` должен увеличиваться;
- `drift_window_size` должен расти до `WINDOW_SIZE` и затем оставаться на этом
  уровне;
- event-time метрики должны присутствовать в `/metrics`.

Для быстрой проверки окна в PowerShell:

```powershell
docker compose -f docker-compose.yaml --profile kafka down
$env:WINDOW_SIZE="20"
$env:MIN_WINDOW_SIZE="5"
$env:PRODUCER_INTERVAL_SECONDS="0.05"
docker compose -f docker-compose.yaml --profile kafka up --build -d
```

После заполнения окна:

```text
drift_window_size 20
```

и значение не должно увеличиваться дальше.

## Проверка Prometheus

Проверить конфигурацию:

```powershell
docker compose -f docker-compose.yaml exec prometheus promtool check config /etc/prometheus/prometheus.yaml
```

Targets:

```text
http://localhost:9090/targets
```

Target `analyzer` должен иметь состояние `UP`.

Временный target `drift-mock-exporter` может быть `DOWN`, если запущен основной
`docker-compose.yaml`, а не `docker-compose-yulia.yaml`.

Prometheus UI:

```text
http://localhost:9090
```

Проверочные запросы:

```promql
drift_events_processed_total
```

```promql
drift_window_size
```

```promql
drift_event_time_lag_seconds
```

End-to-end проверка:

```promql
increase(drift_events_processed_total[1m])
```

При работающем producer результат должен быть больше `0`.

## Grafana

Grafana доступна по адресу:

```text
http://localhost:3000
```

Grafana использует Prometheus datasource из `monitoring/grafana/provisioning/`.
Настройка dashboards находится вне ingestion/exporters.

## Остановка

```powershell
docker compose -f docker-compose.yaml --profile kafka down
Remove-Item Env:WINDOW_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:MIN_WINDOW_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:PRODUCER_INTERVAL_SECONDS -ErrorAction SilentlyContinue
```

## Проверка перед commit

Перед commit проверить diff относительно актуальной `main`:

```powershell
git diff origin/main --name-status
git diff --check origin/main
```

Из существующих общих файлов realtime-ветка изменяет только то, что требуется
для интеграции: `README.md`, `pyproject.toml`, `uv.lock` после `uv lock` и
`monitoring/prometheus/prometheus.yaml`. Остальные изменения относятся к новым
файлам ingestion/exporters/Docker/tests.

Код analyzer/Core, reporting, Grafana, notebooks и данные этой веткой не
изменяются.
