# Data Drift Guardian

Система мониторинга data drift для batch- и realtime-сценариев.

Локальный стенд использует один `docker-compose.yaml` и один конфигурационный
файл Prometheus. Режим запуска выбирается через профиль Docker Compose:

- `mock` — проверка Prometheus и Grafana без Kafka;
- `realtime` — Kafka, analyzer, demo producer, Prometheus и Grafana.

Профили запускаются по отдельности: mock-exporter и analyzer используют один
порт `8000` и общий сетевой alias `drift-exporter`.

## Архитектура realtime

```text
Kafka producer
    ↓
features-stream
    ↓
Kafka consumer
    ↓
KafkaEvent + SchemaChecker
    ↓
непересекающееся окно из WINDOW_SIZE событий
    ↓
pandas.DataFrame
    ↓
EngineAdapter
    ├── основной drift-report
    └── adversarial validation
    ↓
PrometheusExporter
    ↓
Prometheus
    ↓
Grafana
```

Realtime-слой отвечает за Kafka, проверку входных данных, сбор окон,
stream-health, adversarial validation и экспорт результатов в Prometheus.

В текущем состоянии репозитория `src/drift_guardian/engine.py` остаётся
интеграционной заглушкой и читает основной drift-report из
`reports/mock_drift_report.json`. Adversarial validation, stream-health,
Kafka ingestion и Prometheus export выполняются реальным кодом. При общей
интеграции заглушка должна быть заменена вызовом актуального Core API.

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
uv run python -m compileall -q src tests monitoring/mock_exporter
uv run python -m pytest -q
```

Тесты покрывают контракт событий, `SchemaChecker`, reference sampling,
adversarial validation, оконную обработку, Kafka readiness, stream-health,
Prometheus contract и Grafana provisioning.

## Kafka-событие

Пример события:

```json
{
  "event_id": 123,
  "event_time": "2026-09-20T12:00:00Z",
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

## Проверка схемы

`SchemaChecker` поддерживает два режима:

- `check_event()` — проверка одного Kafka-события через pydantic;
- `check_df()` — проверка колонок и pandas dtypes для целого DataFrame.

Признаки на уровне события считаются optional. Отсутствующее значение
материализуется как `None`, чтобы drift-метрики могли учитывать пропуски.
Переданные значения при этом проверяются по типам reference dataset.

Для pandas `category` сравнивается сам тип колонки, а не конкретный набор
категорий. Nullable integer/boolean dtypes (`Int64`, `boolean` и аналогичные)
в текущем контракте не поддерживаются.

## Окна и Kafka offsets

События собираются в полные непересекающиеся окна по `WINDOW_SIZE`.
После успешного анализа полного окна:

1. report экспортируется в Prometheus;
2. увеличивается `drift_analysis_runs_total`;
3. Kafka offsets подтверждаются синхронным commit;
4. окно очищается;
5. начинается следующее окно.

При ошибке анализа commit полного окна не выполняется.

Основные технические метрики:

```text
drift_window_size
drift_current_window_events
drift_events_processed_total
drift_analysis_runs_total
drift_last_analysis_age_seconds
```

Количество событий в текущем окне публикуется напрямую из фактического
состояния `WindowBuffer`:

```promql
drift_current_window_events
```

После добавления принятого события Gauge получает `len(window)`. После
успешного анализа, синхронного commit и очистки окна он сбрасывается в `0`.
Если анализ или commit завершается ошибкой, Gauge сохраняет фактический размер
неочищенного окна.

## Счётчики и выбранный период

Счётчики `drift_events_processed_total`, `drift_analysis_runs_total`,
`drift_out_of_order_events_total` и `drift_late_events_total` считаются с
момента запуска текущего exporter/analyzer. После перезапуска процесса они
начинаются с нуля. Это представление **Since exporter start**.

Для **Over Selected Period** Grafana считает прирост каждого Counter за
выбранный диапазон. `increase()` учитывает сбросы Counter при перезапусках,
`sum()` объединяет временные ряды, а округление выполняется после суммирования:

```promql
round(sum(increase(drift_events_processed_total[$__range]))) or on() vector(-999)
round(sum(increase(drift_analysis_runs_total[$__range]))) or on() vector(-999)
round(sum(increase(drift_out_of_order_events_total[$__range]))) or on() vector(-999)
round(sum(increase(drift_late_events_total[$__range]))) or on() vector(-999)
```

Поэтому значение за выбранный период может быть больше текущего значения
`Since exporter start`, если диапазон включает события до последнего перезапуска
или несколько временных рядов. Служебное значение `-999` отображается в панели
как отсутствие данных.

## Drift report и Prometheus

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

Статусы:

```text
-1 = insufficient_data / not configured
 0 = ok
 1 = warning
 2 = critical
```

Глобальные пороги задаются в `thresholds`. Для отдельной feature можно задать
override в `features.<feature>.thresholds`. При этом
`drift_threshold{metric,level}` продолжает отображать глобальные пороги.

Для обычных drift-метрик используется направление:

```text
warning < critical
```

Для `chi2` экспортируется p-value, поэтому направление обратное:

```text
warning > critical
```

## Reference data

Reference dataset загружается один раз. Для realtime формируется
детерминированная выборка размером до:

```text
REFERENCE_SAMPLE_MULTIPLIER * WINDOW_SIZE
```

По умолчанию `REFERENCE_SAMPLE_MULTIPLIER=10`.

Метаданные reference sample экспортируются отдельно:

```text
drift_reference_profile_info{dataset_name,profile_created_at}
drift_reference_sample_size
```

Исходные CSV/Parquet-файлы размещаются в `data/` и не коммитятся в репозиторий.
Demo reference создаётся автоматически, если файл из `REFERENCE_DATA_PATH`
отсутствует и `GENERATE_DEMO_REFERENCE=true`.

Ручная генерация:

```powershell
uv run python -m drift_guardian.ingestion.demo_reference `
  --output data/demo_reference.csv `
  --rows 10000 `
  --seed 42
```

## Adversarial validation

Adversarial validation запускается после основного анализа полного окна, если:

```text
ENABLE_ADVERSARIAL_VALIDATION=true
```

LightGBM обучается различать reference и current. Датасеты балансируются до
одинакового размера, ROC-AUC рассчитывается по out-of-fold предсказаниям,
feature importance усредняется по CV-фолдам.

Основной Prometheus-контракт AV:

```text
drift_av_status
drift_av_roc_auc
drift_av_last_run_timestamp_seconds
drift_av_dataset_size
drift_av_feature_importance{feature,rank}
```

Дополнительные диагностические метрики:

```text
drift_av_available
drift_av_roc_auc_cv_mean
drift_av_roc_auc_cv_min
drift_av_roc_auc_cv_max
drift_av_roc_auc_cv_std
drift_av_driver_consistency
drift_av_driver_similarity_previous
drift_av_reference_rows
drift_av_current_rows
drift_av_features_evaluated
drift_av_sample_fraction{dataset}
drift_av_threshold{level}
drift_av_top1_importance_share
drift_av_top3_importance_share
```

`drift_av_dataset_size` — размер каждого из двух сбалансированных классов,
использованных AV. `drift_av_driver_consistency` показывает согласованность
importance между CV-фолдами. `drift_av_driver_similarity_previous` сравнивает
importance текущего и предыдущего завершённого AV-запуска.

По умолчанию экспортируется до 10 наиболее важных признаков:

```text
ADVERSARIAL_TOP_FEATURES=10
```

Если доступно меньше признаков, экспортируется фактическое количество.

Пороги AV задаются в `config/config.yaml`:

```yaml
adversarial_validation:
  enabled: true
  thresholds:
    warning: 0.60
    critical: 0.75
```

Статус считается так:

```text
ROC-AUC < warning              -> ok
warning <= ROC-AUC < critical -> warning
ROC-AUC >= critical            -> critical
```

Если AV выключен или результат ещё не получен,
`drift_av_available=0`, а `drift_av_status=-1`.

## Stream health

Публичные stream-метрики:

```text
drift_stream_status
drift_event_time_lag_seconds
drift_window_time_span_seconds
drift_max_event_gap_seconds
drift_invalid_event_time_rate
drift_late_event_rate
drift_late_events_total
drift_out_of_order_events_total
drift_stream_threshold{metric,level}
```

На итоговый `drift_stream_status` влияют только метрики, перечисленные в
`stream_drift` конфигурации. Остальные stream-метрики продолжают экспортироваться
как диагностика.

До первого успешно завершённого окна статус равен `-1`. После этого выбирается
наихудший статус среди настроенных stream-метрик. В текущей конфигурации статус
учитывает lag последнего события и долю late-событий текущего окна. Накопительный
`drift_late_events_total` остаётся диагностическим Counter и не может навсегда
зафиксировать статус в `critical`.

В начале каждого нового окна обнуляются
`drift_window_time_span_seconds`, `drift_max_event_gap_seconds`,
`drift_invalid_event_time_rate` и `drift_late_event_rate`. Эти значения
публикуются сразу после очистки завершённого окна. `drift_event_time_lag_seconds`
обновляется при каждом валидном событии как разница между временем обработки и
его `event_time`.

`drift_late_event_rate` — доля late-событий среди валидных событий текущего
окна. Именно она участвует в расчёте `drift_stream_status`; накопительный
`drift_late_events_total` остаётся диагностическим Counter и не может навсегда
зафиксировать статус `critical`.

В Grafana используется mapping:

```text
0 = Healthy
1 = Degraded
2 = Unhealthy
```

Для AV history:

```text
0 = Healthy
1 = Warning
2 = Critical
```

## Mock profile

Mock используется для проверки Prometheus и Grafana без Kafka:

```powershell
docker compose --profile mock up --build --force-recreate -d
docker compose --profile mock ps -a
```

Mock содержит 20 признаков и prediction, смешанные статусы, изменение metric
values между анализами, demo AV и циклический stream status.

Late-события в mock генерируются из того же потока, из которого рассчитывается
`drift_late_event_rate`: nominal rate составляет `2.5%` в warning-фазе и около
`8.3%` в critical-фазе. `drift_late_events_total` увеличивается только для этих
же событий и не сбрасывается между окнами. Out-of-order события генерируются с
nominal rate `1%` и `4%` соответственно. На границе фаз текущее окно может
содержать события из двух фаз, поэтому его фактическая доля меняется плавно.

После запуска доступны:

- Grafana: `http://localhost:3000`;
- Prometheus: `http://localhost:9090`;
- exporter: `http://localhost:8000/metrics`.

Остановка:

```powershell
docker compose --profile mock down --remove-orphans
```

## Realtime profile

Запуск полного pipeline:

```powershell
docker compose --profile realtime up --build --force-recreate -d
docker compose --profile realtime ps -a
```

Ожидаемые сервисы:

```text
kafka
kafka-init
analyzer
drift-producer
prometheus
grafana
```

`kafka-init` должен завершиться с кодом `0`. Producer стартует после успешного
healthcheck analyzer.

Проверка analyzer:

```powershell
docker compose logs --tail=150 analyzer
```

Проверка Kafka topic:

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

## Grafana и Prometheus

Dashboard загружается автоматически через Grafana provisioning из:

```text
monitoring/grafana/dashboards/drift_guardian.json
```

Ручной import JSON не нужен.

Адреса:

```text
Grafana             http://localhost:3000
Prometheus          http://localhost:9090
Prometheus targets  http://localhost:9090/targets
Exporter            http://localhost:8000/metrics
```

Для history используется относительный диапазон времени и автоматическое
обновление dashboard.

AV row содержит status, ROC-AUC, worst-fold AUC, CV std, согласованность
feature importance между фолдами, similarity с предыдущим окном, top drivers и
технический diagnostic block.

## Основные переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `WINDOW_SIZE` | `1000` | Размер окна анализа |
| `KAFKA_TOPIC` | `features-stream` | Kafka topic |
| `KAFKA_GROUP_ID` | `drift-consumer` | Consumer group |
| `KAFKA_STARTUP_TIMEOUT_SECONDS` | `60` | Таймаут ожидания Kafka topic |
| `KAFKA_STARTUP_RETRY_SECONDS` | `1` | Интервал readiness check |
| `LATE_EVENT_THRESHOLD_SECONDS` | `60` | Порог late-event |
| `REFERENCE_DATA_PATH` | `/app/data/demo_reference.csv` | Reference dataset |
| `GENERATE_DEMO_REFERENCE` | `true` | Автогенерация demo reference |
| `DEMO_REFERENCE_ROWS` | `10000` | Размер demo reference |
| `DEMO_REFERENCE_SEED` | `42` | Seed demo data |
| `REFERENCE_SAMPLE_MULTIPLIER` | `10` | Размер reference sample относительно окна |
| `REFERENCE_SAMPLE_RANDOM_STATE` | `42` | Seed reference sampling |
| `ENABLE_SCHEMA_CHECK` | `true` | Проверка схемы |
| `ENABLE_ADVERSARIAL_VALIDATION` | `true` | Запуск AV |
| `ADVERSARIAL_TOP_FEATURES` | `10` | Число AV importance для экспорта |
| `PRODUCER_INTERVAL_SECONDS` | `0.2` | Интервал demo producer |

## Быстрая end-to-end проверка

```powershell
docker compose --profile realtime up --build --force-recreate -d
docker compose --profile realtime ps -a
docker compose logs --tail=100 analyzer
```

Ключевые runtime-метрики:

```powershell
curl.exe -s http://localhost:8000/metrics |
  Select-String "drift_current_window_events|drift_window_size|drift_events_processed_total|drift_analysis_runs_total|drift_late_event_rate|drift_stream_status|drift_av_roc_auc"
```

После первого полного окна:

- `drift_events_processed_total` должен быть больше `0`;
- `drift_analysis_runs_total` должен быть больше `0`;
- `drift_current_window_events` должен находиться в диапазоне от `0` до
  `drift_window_size` и сбрасываться после полного окна;
- `drift_late_event_rate` должен отражать долю late-событий текущего окна и
  сбрасываться после полного окна;
- `drift_av_available` должен стать `1`, если AV включён;
- `drift_av_feature_importance` должен содержать хотя бы один признак;
- targets Prometheus `prometheus` и `drift-exporter` должны быть `UP`.

## Остановка

Realtime:

```powershell
docker compose --profile realtime down --remove-orphans
```

Mock:

```powershell
docker compose --profile mock down --remove-orphans
```

Для полного сброса Docker volumes можно добавить флаг `--volumes`.

## Проверка перед коммитом

```powershell
uv lock --check
uv sync --frozen
uv run python -m compileall -q src tests monitoring/mock_exporter
uv run python -m pytest -q
git -c core.whitespace=cr-at-eol diff --check
```
