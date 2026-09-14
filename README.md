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
analyzer adapter / DriftMetricsEngine
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

Reference integration:

```text
REFERENCE_DATA_PATH=data/train_transaction_sample.parquet
REFERENCE_SAMPLE_MULTIPLIER=10
REFERENCE_SAMPLE_RANDOM_STATE=42
```

Для parquet `pyarrow` входит в runtime-зависимости проекта.

> В текущем репозитории `train_transaction_sample.parquet` используется только
> для интеграционного smoke: его схема отличается от demo `config.yaml`/producer.
> Пока Core возвращает статический mock-report, это допустимо. Перед включением
> реального drift engine `REFERENCE_DATA_PATH` нужно указать на reference dataset
> с той же feature-схемой, что и текущий поток.

Окно хранит последние `WINDOW_SIZE` валидных событий. После заполнения новое
событие вытесняет самое старое.

Analyzer вызывается только после достижения `MIN_WINDOW_SIZE`:

```python
report = analyzer(window.to_dataframe())
```

`consumer.run()` по-прежнему принимает analyzer с интерфейсом
`Callable[[DataFrame], dict]`. Для текущего Core добавлен
`ingestion/engine_adapter.py`: он один раз загружает reference dataset, берёт
детерминированный sample размером до `10 * WINDOW_SIZE` и адаптирует вызов к
существующей сигнатуре `engine.analyze_dataframe(reference_df, current_df)`.
`engine.py` при этом не меняется.

Стандартный Docker entrypoint теперь собирает этот adapter автоматически через
`REFERENCE_DATA_PATH`, поэтому runtime-цепочка доходит до Core и обратно в
Prometheus exporter. Adapter также публикует `dataset_name`, `sample_size` и
время создания активного reference sample как reference metadata.

## Drift report

Exporter совместим с фактической формой mock `drift_report` из текущего `main`:

```json
{
  "window_size": 1000,
  "overall_status": "critical",
  "active_alerts": 2,
  "features": {
    "age": {
      "type": "numeric",
      "status": "critical",
      "metrics": {
        "psi": 0.31,
        "missing_rate": 0.01,
        "mean_zscore": 3.4
      },
      "alerts": ["psi_critical", "mean_zscore_critical"]
    }
  },
  "prediction": {
    "status": "passed",
    "metrics": {
      "prediction_score_drift": 0.08,
      "positive_prediction_rate": 0.41
    },
    "alerts": []
  }
}
```

Пороговые значения drift-метрик экспортируются как глобальные для типа метрики
через `drift_metric_threshold{metric,level}`, если producer отчёта передал
секцию `thresholds`. Для обратной совместимости exporter также принимает старый
вложенный payload метрики вида `{value, warning, critical, status}`.

## Prometheus metrics

Drift report и состояние анализа:

```text
drift_overall_status
drift_active_alerts
drift_window_size
drift_analysis_runs_total
drift_last_analysis_age_seconds
drift_report_timestamp_seconds
drift_feature_status{feature,type}
drift_feature_psi{feature,type}
drift_feature_missing_rate{feature,type}
drift_feature_mean_zscore{feature,type}
drift_feature_unseen_category_rate{feature,type}
drift_feature_cardinality_ratio{feature,type}
drift_feature_cramer_v_score{feature,type}
drift_feature_active_alerts{feature,type}
drift_feature_alert_info{feature,type,alert}
drift_metric_threshold{metric,level}
drift_prediction_status
drift_prediction_score_drift
drift_prediction_positive_rate
```

Metadata активного reference profile:

```text
drift_reference_profile_info{profile_created_at,dataset_name,...}
drift_reference_sample_size
```

Reference metadata обновляется при загрузке adapter, а не при каждом анализе
окна. Текущая реализация публикует `profile_created_at`, `dataset_name` и
`sample_size`. Когда Core profiler начнёт отдавать собственную metadata, adapter
может передавать её без изменения Prometheus-контракта.

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
drift_late_event_rate
drift_out_of_order_event_rate
drift_late_events_total
drift_out_of_order_events_total
drift_stream_threshold{metric,level}
```

`drift_window_time_span_seconds` справочный и не влияет на статус потока.
`drift_stream_status` вычисляется как худший статус среди настроенных
stream-quality метрик. Пороговые значения пока задаются через временный env
bridge (`STREAM_*_WARNING` / `STREAM_*_CRITICAL`); численные defaults намеренно
не зашиты, потому что они ещё не согласованы в общем config.

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

Тесты ingestion/integration проверяют:

- Kafka event contract;
- timezone-aware `event_time`;
- sliding window и вытеснение старых событий;
- преобразование окна в DataFrame до analyzer;
- адаптацию `analyzer(current_df)` к Core API `analyze_dataframe(reference_df, current_df)`;
- загрузку и детерминированный sampling reference dataset;
- реальный текущий Core `engine.analyze_dataframe` → realtime adapter → exporter;
- stream/time-метрики и rolling window-local quality rates;
- выбор худшего `drift_stream_status` по настроенным thresholds;
- фактический `drift_report` из текущего `main` → Prometheus contract;
- `drift_analysis_runs_total` и metadata reference profile.

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

При старте analyzer сначала ждёт доступности Kafka и появления `KAFKA_TOPIC`, и
только потом создаёт consumer. Это специально убирает гонку запуска, при которой
раньше в логах появлялись временные ошибки `Connection refused` и
`UNKNOWN_TOPIC_OR_PART`, пока broker ещё запускался, а `kafka-init` ещё не успел
создать `features-stream`. По умолчанию ожидание ограничено 60 секундами
(`KAFKA_STARTUP_TIMEOUT_SECONDS`), повторная проверка выполняется раз в секунду
(`KAFKA_STARTUP_RETRY_SECONDS`). Нормальный старт выглядит так:

```text
waiting for Kafka topic=features-stream bootstrap=kafka:19092
Kafka topic ready topic=features-stream bootstrap=kafka:19092
consumer started topic=features-stream
```

После сообщения `Kafka topic ready` в логах analyzer не должно быть стартовых
`%3|...|FAIL|`, `Connection refused` или `UNKNOWN_TOPIC_OR_PART`. Если topic не
появился за timeout, analyzer завершается с явной ошибкой `Kafka startup timeout`
и Docker применяет настроенную restart-policy.

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

## Что именно проверяет Docker smoke

Стандартный `docker-compose.yaml` теперь проверяет полную runtime-цепочку
`Kafka -> consumer -> sliding window -> engine adapter -> Core engine -> drift_report -> Prometheus exporter -> Prometheus -> Grafana`.

Текущий Core `engine.analyze_dataframe` всё ещё является статическим mock и пока
не использует содержимое `reference_df/current_df` для реального расчёта drift.
Поэтому этот smoke подтверждает интеграционный контракт и транспорт данных, но
не статистическую корректность drift-алгоритмов.

## Проверка exporter

Открыть:

```text
http://localhost:8000/metrics
```

При работающем producer:

- `drift_events_processed_total` должен увеличиваться;
- `drift_window_size` должен расти до `WINDOW_SIZE` и затем оставаться на этом
  уровне;
- `drift_analysis_runs_total` после достижения `MIN_WINDOW_SIZE` должен расти;
- `drift_feature_psi{feature="age",type="numeric"}` должен появиться после первого анализа;
- `drift_reference_sample_size` должен быть больше `0`;
- event-time метрики должны присутствовать в `/metrics`.

Для быстрой проверки окна в PowerShell сначала остановить stack, затем задать
переменные **до** команды `up` и пересоздать контейнеры:

```powershell
docker compose -f docker-compose.yaml --profile kafka down
$env:WINDOW_SIZE="20"
$env:MIN_WINDOW_SIZE="5"
$env:ANALYZE_EVERY_N_EVENTS="1"
$env:PRODUCER_INTERVAL_SECONDS="0.05"
docker compose -f docker-compose.yaml --profile kafka up --build --force-recreate -d
```

Проверить, что Compose действительно подставил тестовые значения, можно так:

```powershell
docker compose -f docker-compose.yaml config | Select-String "WINDOW_SIZE|MIN_WINDOW_SIZE|ANALYZE_EVERY_N_EVENTS|PRODUCER_INTERVAL_SECONDS"
```

Если переменные задать после `up` или открыть новый PowerShell, уже созданный
контейнер продолжит работать со старыми значениями.

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

В realtime stack должны быть только два target и оба должны иметь состояние
`UP`:

- `analyzer`;
- `prometheus`.

Для realtime stack используется отдельный
`monitoring/prometheus/prometheus-realtime.yaml`, поэтому Prometheus больше не
пытается scrape-ить отсутствующий `drift-mock-exporter`. Исходный
`monitoring/prometheus/prometheus.yaml` остаётся для mock/Grafana-сценария Юлии
и этим realtime commit не переписывается.

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

## Критерии успешного realtime smoke

Перед commit локальный smoke считается успешным, если одновременно выполняется:

1. `docker compose ... ps` показывает работающие `analyzer`, `prometheus`,
   `grafana`, `kafka` и `drift-producer`; `kafka-init` завершился с кодом `0`.
2. В analyzer log есть `Kafka topic ready` и после него нет startup-ошибок Kafka.
3. `/metrics` показывает рост `drift_events_processed_total` и
   `drift_analysis_runs_total`, а также `drift_feature_psi` и
   `drift_reference_sample_size`.
4. `http://localhost:9090/targets` содержит только `analyzer` и `prometheus`, оба
   `UP`.
5. Grafana открывается и использует Prometheus datasource.

Текущий Core остаётся mock-реализацией, поэтому такой smoke доказывает корректную
интеграцию компонентов, но не статистическую корректность будущего drift engine.

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

Перед commit сначала проверить изменения относительно текущего HEAD своей
ветки:

```powershell
git diff --name-status
git diff --check
```

Для этого commit ожидаются изменения только в realtime/exporter, runtime
зависимостях/compose, тестах и документации. Код analyzer/Core, reporting,
Grafana, notebooks и данные не изменяются.

Отдельно полезно посмотреть расхождение ветки с `origin/main`, но не использовать
его как список файлов текущего commit:

```powershell
git fetch origin
git diff --name-status origin/main...HEAD
```
