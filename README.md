# Data Drift Guardian

Система мониторинга data drift для batch- и realtime-сценариев.

Локальный стенд использует один `docker-compose.yaml` и один конфигурационный
файл Prometheus. Режим запуска выбирается через профиль Docker Compose:

- `mock` — проверка Prometheus и Grafana без Kafka;
- `realtime` — полный pipeline с Kafka, analyzer и realtime exporter.

Профили запускаются **по отдельности**: mock-exporter и analyzer используют один
порт `8000` и общий network alias `drift-exporter`.

## Архитектура realtime

```text
Kafka producer
    ↓
features-stream
    ↓
Kafka consumer
    ↓
SchemaChecker
    ↓
непересекающееся окно из WINDOW_SIZE событий
    ↓
pandas.DataFrame
    ↓
EngineAdapter
    ↓
Core analyzer
    ↓
drift_report
    ↓
PrometheusExporter
    ↓
Prometheus
    ↓
Grafana
```

Realtime-слой отвечает за Kafka, проверку входных данных, сбор окон,
stream-health и экспорт результатов в Prometheus. Расчёт drift-метрик остаётся в
Core, визуализация — в Grafana.

## Требования

- Python 3.13+;
- `uv`;
- Docker Desktop или Docker Engine;
- Docker Compose v2.

## Установка

```powershell
uv lock --check
uv sync --frozen
```

Для зависимостей ноутбуков:

```powershell
uv sync --group notebooks
```

## Тесты

```powershell
uv run python -m compileall -q src tests
uv run python -m unittest discover -s tests -v
```

Тесты покрывают контракт событий, SchemaChecker, reference sampling,
непересекающиеся окна, Kafka readiness, stream-health, Prometheus contract и
Grafana provisioning.

## Kafka-событие

Пример события:

```json
{
  "event_id": 123,
  "event_time": "2026-09-16T10:30:00Z",
  "age": 45,
  "income": 85000,
  "country": "DE",
  "prediction_score": 0.72
}
```

Требования к событию:

- `event_id` — строка или целое число;
- `event_time` — ISO-8601 timestamp с timezone;
- поля признаков — плоские скалярные JSON-значения;
- вложенные `list` и `dict` отклоняются;
- `prediction_score`, если передан, должен быть числом в диапазоне `[0, 1]`.

`event_id` и `event_time` не попадают в DataFrame для drift-анализа.

## Mock-мониторинг

Режим `mock` используется для проверки Prometheus и Grafana без Kafka:

```powershell
docker compose --profile mock up --build -d
docker compose --profile mock ps -a
```

После запуска доступны:

- Grafana: `http://localhost:3000`;
- Prometheus: `http://localhost:9090`;
- exporter: `http://localhost:8000/metrics`.

Dashboard загружается автоматически через Grafana provisioning. Ручной импорт
JSON и дополнительная настройка панелей не нужны.

Если dashboard открылся на сохранённом историческом интервале, выберите
`Last 30 minutes` и обновите страницу.

Остановка:

```powershell
docker compose --profile mock down --remove-orphans
```

## Realtime

Запуск полного pipeline:

```powershell
docker compose --profile realtime up --build -d
docker compose --profile realtime ps -a
```

При нормальном старте `kafka-init` завершается с кодом `0`, а analyzer после
создания топика запускает consumer:

```text
waiting for Kafka topic=features-stream bootstrap=kafka:19092
Kafka topic ready topic=features-stream bootstrap=kafka:19092
consumer started topic=features-stream window_size=1000
```

Проверка analyzer:

```powershell
docker compose logs --tail=100 analyzer
```

Проверка Kafka-топика:

```powershell
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:19092 `
  --describe `
  --topic features-stream
```

Проверка exporter:

```powershell
curl.exe -s http://localhost:8000/metrics
```

Targets Prometheus доступны по адресу `http://localhost:9090/targets`.
`prometheus` и `drift-exporter` должны быть в состоянии `UP`.

## Окна и анализ

События собираются в полные **непересекающиеся** окна по `WINDOW_SIZE`.
Анализ запускается один раз после заполнения окна. После успешного анализа окно
очищается и начинается следующее.

Kafka offsets подтверждаются после успешного анализа полного окна, поэтому
частично накопленное окно не считается завершённым анализом.

Основные метрики:

```text
drift_window_size
drift_events_processed_total
drift_analysis_runs_total
```

Количество уже накопленных событий текущего окна можно вычислить как:

```promql
drift_events_processed_total - drift_analysis_runs_total * drift_window_size
```

## Drift report и Prometheus

Core хранит `drift_report.timestamp` в ISO-8601 UTC. Exporter преобразует его в
Unix timestamp в секундах только при экспорте в Prometheus:

```text
drift_report_timestamp_seconds
```

Основной контракт drift-метрик:

```text
drift_overall_status
drift_active_alerts
drift_report_timestamp_seconds

drift_status_feature{feature,type}
drift_metric_value{feature,type,metric}
drift_status{feature,type,metric}
drift_threshold{metric,level}
```

Глобальные пороги drift-метрик задаются в `thresholds`. Для отдельной фичи
Core может использовать override из `features.<feature>.thresholds`; exporter не
пересчитывает feature-status и принимает его из `drift_report`. Метрика
`drift_threshold{metric,level}` по-прежнему отображает глобальные пороги.

Статусы:

```text
-1 = insufficient_data
 0 = ok
 1 = warning
 2 = critical
```

## Stream health

Публичные stream-метрики:

```text
drift_stream_status
drift_event_time_lag_seconds
drift_window_time_span_seconds
drift_max_event_gap_seconds
drift_invalid_event_time_rate
drift_late_events_total
drift_out_of_order_events_total
drift_stream_threshold{metric,level}
```

`drift_window_time_span_seconds` — справочная метрика. Пороговые значения для
stream-quality берутся из секции `stream_drift` общего `config/config.yaml`. В
итоговый `drift_stream_status` входят только перечисленные там метрики; остальные
продолжают экспортироваться как диагностика, но на статус не влияют.

В текущем конфиге контролируются `drift_event_time_lag_seconds` (`30/120` секунд)
и накопительный `drift_late_events_total` (`10/50` событий). До первого успешно
проанализированного окна статус остаётся `-1`; затем выбирается худший статус по
настроенным метрикам.

`drift_late_events_total` и `drift_out_of_order_events_total` — накопительные
счётчики за всё время работы процесса. `drift_invalid_event_time_rate` относится
к текущему окну и сбрасывается после успешного анализа.

## Reference data и проверка схемы

`EngineAdapter` загружает reference dataset один раз и формирует
детерминированную выборку размером до:

```text
REFERENCE_SAMPLE_MULTIPLIER * WINDOW_SIZE
```

По умолчанию `REFERENCE_SAMPLE_MULTIPLIER=10`.

Перед добавлением события в окно схема признаков проверяется через
`SchemaChecker`. Отсутствующее значение допускается и материализуется как
`None`, чтобы Core мог учитывать `missing_rate`; значение неверного типа
отклоняется.

Метаданные reference sample экспортируются отдельно:

```text
drift_reference_profile_info{profile_created_at,dataset_name}
drift_reference_sample_size
```

## Данные

Исходные CSV/Parquet-файлы не коммитятся в репозиторий. Локальные данные
размещаются в `data/`, которая находится в `.gitignore`.

Для demo reference используется фиксированный seed, поэтому генерация
воспроизводима:

```powershell
uv run python -m drift_guardian.ingestion.demo_reference `
  --output data/demo_reference.csv `
  --rows 10000 `
  --seed 42
```

При realtime-запуске demo reference создаётся автоматически, если файл из
`REFERENCE_DATA_PATH` отсутствует и `GENERATE_DEMO_REFERENCE=true`.

Для собственного reference dataset задайте `REFERENCE_DATA_PATH` и отключите
генерацию demo-данных.

## Конфигурация

Основные переменные окружения:

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `WINDOW_SIZE` | `1000` | Размер окна анализа |
| `KAFKA_TOPIC` | `features-stream` | Kafka-топик |
| `KAFKA_GROUP_ID` | `drift-consumer` | Consumer group |
| `KAFKA_STARTUP_TIMEOUT_SECONDS` | `60` | Таймаут ожидания топика |
| `KAFKA_STARTUP_RETRY_SECONDS` | `1` | Интервал проверки готовности |
| `LATE_EVENT_THRESHOLD_SECONDS` | `60` | Порог опоздания события |
| `REFERENCE_DATA_PATH` | `/app/data/demo_reference.csv` | Reference dataset |
| `GENERATE_DEMO_REFERENCE` | `true` | Создание demo reference |
| `DEMO_REFERENCE_ROWS` | `10000` | Размер demo reference |
| `DEMO_REFERENCE_SEED` | `42` | Seed генератора |
| `REFERENCE_SAMPLE_MULTIPLIER` | `10` | Размер выборки относительно окна |
| `REFERENCE_SAMPLE_RANDOM_STATE` | `42` | Seed выборки |
| `ENABLE_SCHEMA_CHECK` | `true` | Проверка схемы признаков |
| `PRODUCER_INTERVAL_SECONDS` | `0.2` | Интервал demo producer |

Пороги stream-quality задаются в общем YAML-конфиге. Например:

```yaml
stream_drift:
  drift_event_time_lag_seconds:
    warning: 30
    critical: 120
  drift_late_events_total:
    warning: 10
    critical: 50
```

Если метрика не перечислена в `stream_drift`, она не участвует в расчёте
`drift_stream_status`.

## Быстрая end-to-end проверка

```powershell
docker compose --profile realtime up --build --force-recreate -d
docker compose --profile realtime ps -a
docker compose logs --tail=100 analyzer
```

Проверка ключевых метрик:

```powershell
curl.exe -s http://localhost:8000/metrics |
  Select-String "drift_events_processed_total|drift_analysis_runs_total|drift_metric_value|drift_reference_sample_size|drift_stream_status"
```

После заполнения первого окна:

- `drift_events_processed_total` должен быть больше `0`;
- `drift_analysis_runs_total` должен быть больше `0`;
- `drift_metric_value` должен содержать результаты отчёта;
- `drift_reference_sample_size` должен быть больше `0`;
- targets Prometheus `prometheus` и `drift-exporter` должны быть `UP`;
- Grafana должна отображать данные из настроенного datasource Prometheus.

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Prometheus targets: `http://localhost:9090/targets`

## Остановка и полный сброс Docker

Обычная остановка realtime-стека удаляет контейнеры и сеть проекта, но сохраняет
Docker volumes и образы:

```powershell
docker compose --profile realtime down --remove-orphans
```

Если нужен полностью чистый запуск без сохранённых данных Prometheus/Grafana,
удалите также volumes и images проекта:

```powershell
docker compose --profile realtime down `
  --volumes `
  --rmi all `
  --remove-orphans
```

Для mock-профиля используется та же схема:

```powershell
docker compose --profile mock down `
  --volumes `
  --rmi all `
  --remove-orphans
```

Проверить оставшиеся ресурсы можно командами:

```powershell
docker ps -a
docker volume ls
docker image ls
```

`down --volumes` удаляет Docker volumes, но не удаляет файлы из bind mounts,
то есть каталоги, примонтированные напрямую с хоста. Такие данные остаются на
диске и при необходимости удаляются отдельно вручную.

После полного сброса чистая пересборка выполняется так:

```powershell
docker compose --profile realtime build --no-cache
docker compose --profile realtime up -d
```

## Проверка перед коммитом

```powershell
uv lock --check
uv sync --frozen
uv run python -m compileall -q src tests
uv run python -m unittest discover -s tests -v
git diff --check
```
