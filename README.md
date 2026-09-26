# Data Drift Guardian

Data Drift Guardian — сервис мониторинга **data drift** для batch/offline- и realtime-сценариев. Проект строит reference-профиль, считает feature/prediction drift, выполняет adversarial validation (AV), принимает realtime-события из Kafka, экспортирует метрики в Prometheus и визуализирует состояние в Grafana с provisioning alert rules.

## Возможности системы мониторинга

- drift-анализ numeric и categorical features;
- отдельный monitoring prediction drift;
- global thresholds и feature-level overrides;
- автогенерация конфигурации и автокалибровка thresholds по reference-данным;
- adversarial validation на LightGBM с CV diagnostics и feature importance;
- offline HTML report;
- realtime ingestion из Kafka полными непересекающимися окнами;
- stream-health по event-time метрикам;
- Prometheus exporter;
- Grafana dashboard и provisioning alerting;
- локальный demo-режим с Kafka и producer, умеющим создавать контролируемый drift;
- интеграционные тесты с внешней Kafka через `testcontainers`.

## Архитектура

### Realtime

```text
reference dataset + config.yaml
            │
            ▼
      OfflineWrapper / Core
            ▲
            │ current window
            │
external Kafka OR local demo Kafka
            │
            ▼
      Kafka consumer
            │
            ▼
 KafkaEvent → WindowBuffer
            │
      full WINDOW_SIZE
            │
            ▼
       pandas.DataFrame
            │
            ├── feature drift
            ├── prediction drift
            └── scheduled AV
            │
            ▼
    PrometheusExporter
            │
            ▼
       Prometheus
            │
            ▼
 Grafana dashboard + alerting
```

Realtime-слой отвечает за Kafka ingestion, оконную обработку, stream-health и Prometheus export. Расчёт drift-метрик, thresholds и AV выполняется через единый analyzer/Core.

### Offline

```text
reference_df + current_df
          │
          ▼
    OfflineWrapper
      ├── analyze_df() → drift report
      └── run_av()     → ROC AUC + feature importance
          │
          ▼
 generate_html_report()
          │
          ▼
      HTML report
```

## Структура репозитория

```text
config/
  config.yaml                         # runtime config

data/                                 # локальные reference-данные (gitignored)

src/drift_guardian/
  analyzer/                            # Core drift engine, metrics, AV, offline wrapper
  config_handler/                      # config parser + auto config builder
  data_quality_checker/                # schema validation
  exporters/                           # Prometheus exporter
  ingestion/                           # Kafka consumer, windows, stream-health, runtime
  profiler/                            # reference profiling
  reporting/                           # offline HTML report

monitoring/
  grafana/                             # dashboard + provisioning
  prometheus/                          # scrape config
  mock_exporter/                       # mock monitoring contract

tools/
  build_config.py                      # генерация config/config.yaml
  get_demo_data.py                     # загрузка reference dataset по URL
  demo_producer/                       # локальный Kafka producer + drift scenario

notebooks/
  02_offline_report.ipynb              # пример offline workflow

tests/
  integration/                         # Kafka integration tests via testcontainers
```

Проект использует `src` layout. Импорты выполняются через `drift_guardian...`, а не через `src.drift_guardian...`.

## Требования

- Python `>= 3.13`;
- `uv`;
- Docker Engine / Docker Desktop;
- Docker Compose v2;
- для integration tests — работающий Docker daemon.

## Установка

Из корня репозитория:

```powershell
uv sync --frozen
```

Для ноутбуков дополнительно:

```powershell
uv sync --frozen --group notebooks
```

## Быстрый старт: локальный realtime demo

Локальный demo поднимает:

- `analyzer`;
- `kafka`;
- `drift-producer`;
- `prometheus`;
- `grafana`.

### 1. Подготовить reference dataset

Realtime analyzer по умолчанию ожидает:

```text
data/reference.csv
```

В Docker это `/app/data/reference.csv`.

Папка `data/` исключена из Git, поэтому в чистом checkout reference-файла может не быть. Подготовьте его одним из способов:

- положите свой CSV в `data/reference.csv`;
- либо скачайте dataset по URL через `tools/get_demo_data.py` — см. раздел [Загрузка reference dataset по URL](#загрузка-reference-dataset-по-url).

Например, если CSV уже лежит локально в другой папке:

```powershell
New-Item -ItemType Directory -Force .\data | Out-Null
Copy-Item C:\path\to\reference.csv .\data\reference.csv
```

> `data/reference.csv` должен существовать **до запуска analyzer**. Автоматическая генерация reference внутри realtime runtime отключена.

### 2. Проверить `config/config.yaml`

Checked-in config сейчас мониторит:

```text
age
income
country
prediction_score (prediction drift)
```

При использовании другого dataset сначала сгенерируйте/обновите config через `tools/build_config.py`.

### 3. Настроить demo drift

Сценарий локального producer находится в:

```text
tools/demo_producer/drift_config.yaml
```

Поддерживаемые типы drift:

- `shift` — аддитивный сдвиг numeric feature;
- `scale` — изменение масштаба numeric feature;
- `noise` — добавление Gaussian noise;
- `categorical_swap` — замена части значений categorical feature.

Для каждого правила используются:

```yaml
start_step: 1000   # с какого сообщения начинается drift
ramp_steps: 1000   # за сколько сообщений он доходит до полной силы
```

Важно: правила применяются **только если имя feature есть в reference dataset**. Если `drift_config.yaml` содержит `feature_1`, а dataset содержит `age`, это правило будет пропущено.

Пример для `age / income / country`:

```yaml
features:
  age:
    type: shift
    magnitude: 10
    start_step: 1000
    ramp_steps: 1000

  income:
    type: scale
    magnitude: 0.35
    start_step: 1000
    ramp_steps: 1000

  country:
    type: categorical_swap
    probability: 0.5
    target_category: drift_country
    start_step: 1500
    ramp_steps: 500
```

Такое расписание удобно для демонстрации: первое окно остаётся близким к reference, затем drift постепенно нарастает.

### 4. Запустить локальную Kafka + realtime

```powershell
docker compose --profile realtime --profile local-kafka up -d --build
```

Проверка:

```powershell
docker compose --profile realtime --profile local-kafka ps -a
docker compose --profile realtime --profile local-kafka logs --tail=200 analyzer
docker compose --profile realtime --profile local-kafka logs --tail=100 drift-producer
```

Endpoints:

```text
Analyzer metrics: http://localhost:8000/metrics
Prometheus:       http://localhost:9090
Grafana:          http://localhost:3000
```

В PowerShell:

```powershell
curl.exe -s http://localhost:8000/metrics | Select-String "drift_"
```

## Realtime с внешней Kafka

`local-kafka` — только demo profile. Для production-like сценария analyzer подключается непосредственно к внешнему Kafka cluster.

В `.env` задайте как минимум:

```env
KAFKA_BOOTSTRAP_SERVERS=broker1:9092
KAFKA_TOPIC=features-stream
KAFKA_GROUP_ID=drift-consumer
```

Затем запускайте **только** realtime profile:

```powershell
docker compose --profile realtime up -d --build
```

В этом режиме локальные `kafka` и `drift-producer` не поднимаются.

### Требования к внешней Kafka

- broker должен быть доступен из контейнера `analyzer`;
- topic из `KAFKA_TOPIC` должен быть подготовлен владельцем Kafka cluster; проект не содержит `kafka-init` и сам явно topic не создаёт;
- если на стороне broker разрешён auto-create, topic может появиться по политике самого Kafka, но README не предполагает это как обязательное поведение;
- analyzer ждёт доступности topic до `KAFKA_STARTUP_TIMEOUT_SECONDS` (по умолчанию `60` секунд);
- текущий consumer параметризует только `bootstrap.servers`, `group.id` и topic. SASL/SSL credentials и дополнительные Kafka client properties через Compose пока не прокинуты, поэтому защищённый внешний cluster потребует отдельной конфигурации клиента.

Полезные параметры:

```env
KAFKA_STARTUP_TIMEOUT_SECONDS=60
KAFKA_STARTUP_RETRY_SECONDS=1
WINDOW_SIZE=1000
LATE_EVENT_THRESHOLD_SECONDS=60
```

## Mock monitoring mode

Mock profile нужен для проверки Grafana/Prometheus/alerting без Kafka и Core:

```powershell
docker compose --profile mock up -d --build
```

Проверка:

```powershell
docker compose --profile mock ps -a
curl.exe -s http://localhost:8000/metrics | Select-String "drift_"
```

`drift-mock-exporter` и `analyzer` оба публикуют host port `8000`, поэтому mock и realtime exporter одновременно запускать нельзя.

Перед переходом из mock в realtime:

```powershell
docker compose --profile mock stop drift-mock-exporter
docker compose --profile mock rm -f drift-mock-exporter
```

Перед возвратом в mock из внешнего-Kafka realtime режима:

```powershell
docker compose --profile realtime stop analyzer
docker compose --profile realtime rm -f analyzer
```

Если был запущен полный локальный demo (`realtime + local-kafka`):

```powershell
docker compose --profile realtime --profile local-kafka stop drift-producer analyzer kafka
docker compose --profile realtime --profile local-kafka rm -f drift-producer analyzer kafka
```

Prometheus и Grafana не принадлежат profile и могут оставаться поднятыми между режимами.

## Docker profiles

| Profile | Сервисы | Назначение |
|---|---|---|
| без profile | `prometheus`, `grafana` | общая monitoring-инфраструктура |
| `mock` | `drift-mock-exporter` | mock Prometheus contract |
| `realtime` | `analyzer` | Kafka consumer + Core + exporter |
| `local-kafka` | `kafka`, `drift-producer` | локальный demo broker и producer |
| `realtime + local-kafka` | analyzer + local Kafka + producer | полный локальный realtime demo |

Полный останов проекта:

```powershell
docker compose --profile mock --profile realtime --profile local-kafka down --remove-orphans
```

Полный сброс вместе с volume Grafana:

```powershell
docker compose --profile mock --profile realtime --profile local-kafka down --remove-orphans --volumes
```

## Reference dataset

### Локальный файл

Analyzer принимает `.csv` и `.parquet` через `REFERENCE_DATA_PATH`.

Compose по умолчанию использует:

```env
REFERENCE_DATA_PATH=/app/data/reference.csv
```

Папка `./data` монтируется целиком в `/app/data`, поэтому для Parquet можно задать, например:

```env
REFERENCE_DATA_PATH=/app/data/reference.parquet
```

Для локального `drift-producer` текущий Compose монтирует именно `./data/reference.csv`, поэтому demo producer из Compose ожидает CSV с этим именем.

### Загрузка reference dataset по URL

`tools/get_demo_data.py` читает `DATASET_URL` из локального `.env`, при необходимости создаёт папку `data/` и сохраняет результат как:

```text
data/reference.csv
```

Пример `.env`:

```env
DATASET_URL=https://example.org/reference.csv
```

Поддерживается и Google Drive URL:

```env
DATASET_URL=https://drive.google.com/file/d/<FILE_ID>/view
```

Запуск:

```powershell
uv run python tools/get_demo_data.py
```

Скрипт также распознаёт ZIP/GZIP и извлекает данные в `data/reference.csv`.

## Генерация `config/config.yaml`

Подробный генератор находится в:

```text
tools/build_config.py
```

Перед запуском проверьте в начале файла:

```python
DATA_PATH = "data/reference.csv"
OUTPUT_PATH = "config/config.yaml"
```

После настройки:

```powershell
uv run python tools/build_config.py
```

Скрипт:

- определяет numeric/categorical features;
- умеет исключать технические/time columns;
- выбирает набор metrics;
- строит global/local thresholds;
- по умолчанию калибрует thresholds bootstrap-окнами reference dataset;
- добавляет базовые `stream_drift` thresholds;
- может включить prediction monitoring;
- может включить AV;
- валидирует итоговую конфигурацию перед записью.

### Важные defaults генератора

В `tools/build_config.py` по умолчанию:

```text
prediction_enabled = False
adversarial_validation.enabled = False
auto_thresholds.enabled = True
stream_drift = event_time_lag + late_event_rate
```

Поэтому запуск генератора **без редактирования OPTIONS** перезапишет `config/config.yaml` с выключенными prediction и AV.

Чтобы включить prediction:

```python
prediction_enabled=True,
prediction_score_column="prediction_score",
```

Чтобы включить AV:

```python
adversarial_validation=AdversarialValidationBuildOptions(
    enabled=True,
    interval_minutes=30,
    max_samples=50_000,
    n_splits=5,
),
```

## Drift metrics и thresholds

### Поддерживаемые feature metrics

| Metric | Numeric | Categorical | Примечание |
|---|:---:|:---:|---|
| `missing_rate` | ✓ | ✓ | доля пропусков |
| `psi` | ✓ | ✓ | Population Stability Index |
| `js_divergence` | ✓ | ✓ | Jensen-Shannon divergence |
| `wasserstein_distance` | ✓ | — | зависит от масштаба feature |
| `kstest` | ✓ | — | KS D-statistic, не p-value |
| `unseen_category_rate` | — | ✓ | доля новых категорий |
| `cardinality_ratio` | — | ✓ | изменение cardinality |
| `chi2` | — | ✓ | **chi-square p-value** |
| `cramer_v` | — | ✓ | Cramér's V |
| `category_churn` | — | ✓ | изменение состава категорий |

### Global и local thresholds

Глобальные thresholds:

```yaml
thresholds:
  psi:
    warning: 0.10
    critical: 0.25
```

Feature override:

```yaml
features:
  age:
    type: numeric
    metrics: [missing_rate, psi]
    thresholds:
      psi:
        warning: 0.05
        critical: 0.12
```

Приоритет:

```text
feature local threshold > global threshold
```

Core формирует resolved thresholds, exporter публикует:

```text
drift_threshold{metric,level}
drift_resolved_threshold{feature,type,metric,level}
```

Для большинства metrics используется направление:

```text
warning < critical
```

`chi2` возвращает p-value, поэтому для него **меньше = хуже** и направление обратное:

```text
warning > critical
```

например:

```yaml
chi2:
  warning: 0.05
  critical: 0.01
```

## Prediction drift

Prediction monitoring задаётся отдельным блоком:

```yaml
prediction_metrics:
  enabled: true
  score_column: prediction_score
  type: numeric
  metrics:
    - psi
```

Prediction не учитывается как обычная input feature в `active_alerts`: `drift_active_alerts` — количество input features с critical status. Prediction status экспортируется отдельно через series с `feature="prediction"`.

Если prediction column настроена, realtime AV исключает её из feature set AV: prediction drift и multivariate feature drift не смешиваются в одном driver ranking.

## Realtime windows и Kafka offsets

Размер окна задаётся:

```env
WINDOW_SIZE=1000
```

Consumer формирует **полные непересекающиеся окна**. Partial window не анализируется.

Для валидного события:

1. событие добавляется в `WindowBuffer`;
2. обновляются stream metrics;
3. Kafka offset сохраняется локально через `store_offsets`;
4. после заполнения окна выполняется Core analysis;
5. report экспортируется в Prometheus;
6. при успешном анализе увеличивается `drift_analysis_runs_total`;
7. после попытки анализа выполняется synchronous Kafka commit;
8. окно очищается и начинается следующее.

### Поведение при плохом окне

Если анализ полного окна падает, consumer **не завершается**. Ошибка логируется, окно отбрасывается, offset подтверждается и обработка продолжается со следующего окна.

Это сделано для resilience realtime consumer: одно некорректное окно не должно зациклить сервис на одном и том же наборе сообщений.

Некорректные отдельные Kafka messages (invalid JSON/schema/event time) также пропускаются и не добавляются в analysis window.

## Stream health

Экспортируются:

```text
drift_event_time_lag_seconds
drift_window_time_span_seconds
drift_max_event_gap_seconds
drift_invalid_event_time_rate
drift_late_event_rate

drift_late_events_total
drift_out_of_order_events_total
```

`drift_stream_status` вычисляется только по stream metrics, для которых в `stream_drift` заданы thresholds.

Если thresholds не настроены, status остаётся:

```text
-1 = insufficient_data / not configured
```

Current checked-in config использует:

```yaml
stream_drift:
  drift_event_time_lag_seconds:
    warning: 30
    critical: 120
  drift_late_event_rate:
    warning: 0.01
    critical: 0.05
```

`drift_late_event_rate` — window-local rate. `drift_late_events_total` и `drift_out_of_order_events_total` — lifetime counters и не используются как текущий health signal.

## Adversarial validation

AV включается через runtime config:

```yaml
adversarial_validation:
  enabled: true
  interval_minutes: 30
  max_samples: 50000
  n_splits: 5
  random_state: 42
  missing_category: "__missing__"
```

### Когда запускается AV

- первый AV запускается на **первом полном analysis window**, если AV включён и обе выборки, reference и current, содержат достаточно строк для `n_splits`;
- затем AV запускается не чаще чем раз в `interval_minutes`;
- фактический повторный запуск происходит на первом полном окне после истечения интервала;
- обычный drift analysis продолжает выполняться на каждом полном окне независимо от AV schedule.

То есть при `WINDOW_SIZE=1000` и `interval_minutes=30`:

```text
window #1       → drift + AV #1
window #2..N    → drift, последний AV snapshot сохраняется
>= 30 минут     → ближайшее полное окно → drift + AV #2
```

`drift_av_available` — **результат**, а не переключатель AV:

```text
0 = успешного AV snapshot ещё нет
1 = AV snapshot опубликован
```

До первого успешного AV diagnostic gauges могут содержать `-1`, включая:

```text
drift_av_last_run_timestamp_seconds
```

### AV monitoring metrics

```text
drift_av_status
drift_av_available
drift_av_roc_auc
drift_av_roc_auc_cv_mean
drift_av_roc_auc_cv_min
drift_av_roc_auc_cv_max
drift_av_roc_auc_cv_std
drift_av_driver_consistency
drift_av_driver_similarity_previous
drift_av_feature_importance{feature,rank}
drift_av_last_run_timestamp_seconds
drift_av_reference_rows
drift_av_current_rows
drift_av_features_evaluated
```

`drift_av_driver_consistency` — mean pairwise cosine similarity feature-importance vectors между CV folds одного AV run.

`drift_av_driver_similarity_previous` — cosine similarity feature importance текущего AV к предыдущему **завершённому AV**. На первом AV предыдущего snapshot ещё нет, поэтому в Grafana отображается `–`; метрика становится содержательной со второго AV run и не обновляется на обычных drift windows.

Monitoring thresholds для ROC AUC задаются env-переменными analyzer:

```env
AV_WARNING_THRESHOLD=0.60
AV_CRITICAL_THRESHOLD=0.75
```

## Prometheus contract

Основные drift series:

```text
drift_overall_status
drift_active_alerts
drift_status_feature{feature,type}
drift_metric_value{feature,type,metric}
drift_status{feature,type,metric}
drift_threshold{metric,level}
drift_resolved_threshold{feature,type,metric,level}
```

Window/runtime state:

```text
drift_window_size
drift_current_window_events
drift_events_processed_total
drift_analysis_runs_total
drift_last_analysis_age_seconds
drift_report_timestamp_seconds
drift_stream_status
```

Status codes:

```text
-1 = insufficient_data / not configured
 0 = ok
 1 = warning
 2 = critical
```

Prometheus scrape config содержит два отдельных targets:

```text
analyzer:8000
mock exporter:8000
```

Оба relabel'ятся в публичный:

```text
job="drift-exporter"
```

Источники различаются стандартным label `instance`; custom `mode` label не используется.

## Grafana и alerting

Grafana dashboard:

```text
monitoring/grafana/dashboards/drift_guardian.json
```

Provisioning:

```text
monitoring/grafana/provisioning/
  datasources/
  dashboards/
  alerting/
    alert-rules.yaml
    contact-points.yaml
    mute-timings.yaml
    policies.yaml
    templates.yaml
```

Alerting использует streak metrics, чтобы не отправлять notification по единичному всплеску:

```text
drift_overall_status_streak
drift_status_feature_streak
drift_status_streak
drift_av_status_streak
```

Для input/prediction drift alert rule требует status `warning` или `critical` **4 последовательных analysis windows**.

Для AV streak увеличивается только при новом опубликованном AV result; обычные drift windows между AV runs не являются новыми AV observations.

Отдельного Grafana alert rule для `stream_status` сейчас нет.

### Telegram

Compose передаёт в Grafana переменную:

```env
TELEGRAM_BOT_TOKEN=...
```

Текущий `contact-points.yaml` ссылается на `$TELEGRAM_BOT_TOKEN`, поэтому реальное значение хранится только локально в `.env`/окружении. Не заменяйте эту ссылку literal-token'ом в Git. Если token когда-либо был опубликован, его следует перевыпустить.

## Offline analysis и HTML report

Offline workflow строится через `OfflineWrapper` и HTML renderer из `drift_guardian.reporting`.

Минимальный Python API:

```python
import pandas as pd

from drift_guardian.analyzer.offline.offline_mode import OfflineWrapper
from drift_guardian.config_handler.auto_config_builder import ConfigBuildOptions
from drift_guardian.reporting import display_html_report, generate_html_report

reference_df = pd.read_csv("data/reference.csv")
current_df = pd.read_csv("data/current.csv")

config_options = ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
)

analyzer = OfflineWrapper(
    reference_df=reference_df,
    config_options=config_options,
)

report = analyzer.analyze_df(current_df)
av_report = analyzer.run_av(
    current_df,
    prediction_col="prediction_score",
)

generated_html = generate_html_report(
    report=report,
    av_report=av_report,
    dataset_name="Offline drift demo",
    output_path="reports/offline_drift_report.html",
)
```

`prediction_col="prediction_score"` исключает prediction column из AV driver ranking: prediction drift рассчитывается отдельно и не смешивается с multivariate feature drift.

Текущий HTML report содержит:

- сводку `Input Data Drift`;
- feature-level таблицы по типам признаков;
- отдельный блок `Prediction Drift` с status-card и карточками prediction metrics;
- `Adversarial Validation` с ROC AUC и top-10 feature importance, если передан `av_report`;
- warning/critical thresholds рядом со значениями metrics;
- подпись `χ² p-value` для `chi2`, чтобы значение `0`/малый p-value не воспринималось как обычная метрика направления «больше = хуже».

В notebook готовый HTML можно отобразить через:

```python
display_html_report(generated_html, height=1300)
```

Подробный runnable example находится в:

```text
notebooks/02_offline_report.ipynb
```

### CLI для сохранённого JSON report

В `pyproject.toml` зарегистрирован console entry point:

```text
drift-guardian-report
```

После `uv sync` можно сгенерировать HTML из сохранённого JSON report:

```powershell
uv run drift-guardian-report reports/mock_drift_report.json reports/offline_drift_report.html --dataset-name "Offline drift demo"
```

CLI принимает готовый drift report из JSON и не имеет отдельного аргумента для `av_report`. Если в HTML требуется секция Adversarial Validation, используйте Python API `generate_html_report(..., av_report=...)` или notebook.

## Проверки и тесты

### Быстрые unit tests без Docker

```powershell
uv run pytest -q --ignore=tests/integration
```

### Integration tests с Kafka

Integration suite поднимает `apache/kafka:4.3.1` через `testcontainers` и проверяет сценарий внешнего Kafka broker.

Docker должен быть запущен:

```powershell
uv run pytest -q -m integration
```

Проверяются, в частности:

- Kafka roundtrip;
- ожидание broker/topic;
- формирование полных окон;
- отсутствие анализа partial window;
- non-overlapping windows;
- offset commit;
- восстановление после bad window;
- invalid messages/event time;
- stream metrics;
- AV disabled/enabled behavior;
- reconnect к Kafka.

Полный suite:

```powershell
uv run pytest -q
```

Дополнительная syntax-проверка:

```powershell
uv run python -m compileall -q src tests monitoring/mock_exporter
```

## Диагностика

### Analyzer не стартует: `reference dataset not found`

Проверьте:

```powershell
Test-Path .\data\reference.csv
```

Если файла нет — скопируйте локальный example либо используйте `tools/get_demo_data.py`.

### Analyzer ждёт Kafka topic

Логи:

```powershell
docker compose --profile realtime logs --tail=200 analyzer
```

Проверьте:

- `KAFKA_BOOTSTRAP_SERVERS`;
- `KAFKA_TOPIC`;
- network access из контейнера;
- существование topic во внешнем cluster.

### `drift_av_available = 0`

Это означает, что успешный AV snapshot пока не опубликован. Проверьте:

```text
adversarial_validation.enabled
adversarial_validation.interval_minutes
WINDOW_SIZE >= n_splits
reference sample size >= n_splits
analyzer logs
```

Первый AV при корректной конфигурации планируется уже на первом полном окне.

### `drift_av_last_run_timestamp_seconds = -1`

До первого успешного AV это ожидаемо. Если `drift_av_available = 1`, а timestamp остаётся `-1`, это уже повод проверить exporter/analyzer logs.

### PSI меняется, но остаётся `OK`

Это нормально, если current window статистически близок к reference. Для демонстрации переходов `OK → Warning → Critical` настройте `tools/demo_producer/drift_config.yaml`, чтобы producer действительно менял распределение features, присутствующих в dataset.

## Переменные окружения

Основные параметры realtime:

| Переменная | Default | Назначение |
|---|---:|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:19092` | Kafka bootstrap servers |
| `KAFKA_TOPIC` | `features-stream` | input topic |
| `KAFKA_GROUP_ID` | `drift-consumer` | consumer group |
| `KAFKA_STARTUP_TIMEOUT_SECONDS` | `60` | timeout ожидания Kafka/topic |
| `KAFKA_STARTUP_RETRY_SECONDS` | `1` | retry interval |
| `WINDOW_SIZE` | `1000` | размер analysis window |
| `LATE_EVENT_THRESHOLD_SECONDS` | `60` | event считается late после этого lag |
| `REFERENCE_DATA_PATH` | `/app/data/reference.csv` | reference dataset внутри analyzer |
| `ADVERSARIAL_TOP_FEATURES` | `10` | число AV drivers в exporter |
| `AV_WARNING_THRESHOLD` | `0.60` | AV warning ROC AUC |
| `AV_CRITICAL_THRESHOLD` | `0.75` | AV critical ROC AUC |
| `PRODUCER_INTERVAL_SECONDS` | `0.2` | частота local demo producer |
| `PRODUCER_RANDOM_SEED` | `42` | seed local demo producer |
| `DATASET_URL` | — | URL для `tools/get_demo_data.py` |

## Код-стиль и архитектурные ограничения

- imports через `drift_guardian...`;
- один config parser — `src/drift_guardian/config_handler/parse_config.py`;
- realtime не реализует второй Core и второй AV;
- feature/prediction drift и AV выполняются через `OfflineWrapper` / `DriftMetricsEngine`;
- public functions/classes должны иметь понятные type hints и docstrings;
- комментарии должны описывать контракт/причину решения, а не дублировать код.

## Безопасность и локальные файлы

`.gitignore` исключает новые:

```text
.env
.env.*
data/
reports/*.html
.venv/
__pycache__/
```

## Ни в коем случае не стоит хранить в Git:

- Telegram bot token;
- credentials внешней Kafka;
- production datasets;
- приватные URLs/ключи доступа.

Если какой-то из secret уже попал в commit/history, простой перенос в `.env` недостаточен — secret нужно отозвать/перевыпустить.
