# Ingestion and Prometheus

Зона ответственности блока:

```text
Kafka producer
    ↓
Kafka topic
    ↓
consumer
    ↓
sliding window → pandas.DataFrame
    ↓
DriftMetricsEngine.analyze_dataframe(current_df)
    ↓
drift_report
    ↓
PrometheusExporter
    ↓
Prometheus
```

Grafana, drift-метрики и alerting находятся вне ingestion/exporters.

## Структура

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

monitoring/prometheus/prometheus.yaml
docker-compose.yaml
```

## Kafka event

Обязательные поля:

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

`event_id` — строка или целое число. `event_time` должен содержать timezone.
Остальные поля должны быть плоскими JSON scalar-значениями.

`event_id` и `event_time` не попадают в DataFrame. `prediction_score`
остаётся колонкой DataFrame для анализа prediction drift.

## Sliding window

По умолчанию:

```text
WINDOW_SIZE=1000
MIN_WINDOW_SIZE=300
ANALYZE_EVERY_N_EVENTS=1
```

Окно собирается в `pandas.DataFrame` до вызова analyzer. Ingestion ожидает
callable с интерфейсом:

```python
def analyze_dataframe(current_df: pandas.DataFrame) -> dict:
    ...
```

Пока `DriftMetricsEngine` не находится в main, Docker entrypoint запускает
consumer без analyzer. Kafka, окно и stream-метрики работают независимо.
После появления engine в integration layer нужно передать
`engine.analyze_dataframe` в `drift_guardian.ingestion.consumer.run()`.

## Drift report → Prometheus

Exporter принимает новый формат report, где каждая drift-метрика содержит
`value`, `warning`, `critical` и `status`.

Основные series:

```text
drift_overall_status
drift_active_alerts
drift_report_timestamp_seconds
drift_metric_value{feature,metric}
drift_threshold{feature,metric,level}
drift_status{feature,metric}
drift_status_feature{feature}
```

Prediction экспортируется как `feature="prediction"`.

Stream series:

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

## Локальные тесты

```bash
uv lock
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src tests
```

## Docker smoke test

Kafka и demo producer включены отдельным profile:

```bash
docker compose -f docker-compose.yaml --profile kafka up --build -d
docker compose -f docker-compose.yaml ps
```

Проверить topic:

```bash
docker compose -f docker-compose.yaml exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --list
```

Ожидается `features-stream`.

Проверить exporter:

```text
http://localhost:8000/metrics
```

`drift_events_processed_total` должен расти, `drift_window_size` — расти до
`WINDOW_SIZE`.

Проверить Prometheus:

```text
http://localhost:9090/targets
```

Target `analyzer` должен быть `UP`. Target временного `drift-mock-exporter`
может быть `DOWN`, если запущен основной stack, а не `docker-compose-yulia`.

PromQL:

```promql
increase(drift_events_processed_total[1m])
```

Результат при работающем producer должен быть больше нуля.

Grafana:

```text
http://localhost:3000
```

Остановить stack:

```bash
docker compose -f docker-compose.yaml --profile kafka down
```
