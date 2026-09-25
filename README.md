# Data Drift Guardian

Data Drift Guardian — сервис мониторинга data drift для batch- и realtime-сценариев. Realtime-контур принимает события из Kafka, формирует полные окна, передаёт их в единый Core анализа и публикует результаты в Prometheus. Grafana используется для визуализации и alerting.

## Архитектура

```text
Producer
  ↓
Kafka (features-stream)
  ↓
Kafka consumer
  ↓
KafkaEvent → WindowBuffer
  ↓
полное окно WINDOW_SIZE
  ↓
pandas.DataFrame
  ↓
OfflineWrapper / DriftMetricsEngine
  ├─ SchemaChecker.check_df
  ├─ feature/prediction drift
  └─ adversarial validation
  ↓
PrometheusExporter
  ↓
Prometheus
  ↓
Grafana dashboard + alerting + Telegram
```

Realtime-слой отвечает за ingestion, оконную обработку, stream-health и экспорт. Feature/prediction drift, разрешение global/local thresholds и adversarial validation выполняются через Core.

## Структура ключевых компонентов

```text
config/config.yaml
src/drift_guardian/config_handler/        # единый parser и config models
src/drift_guardian/analyzer/              # Core drift engine и AV
src/drift_guardian/data_quality_checker/  # schema validation
src/drift_guardian/ingestion/             # Kafka, окна и realtime runtime
src/drift_guardian/exporters/             # Prometheus contract
monitoring/grafana/                       # dashboard и provisioning
monitoring/mock_exporter/                 # mock monitoring contract
```

Python-пакет использует `src` layout. Импорты выполняются через `drift_guardian...`; префикс `src.` в runtime-коде не используется.

## Требования

- Python 3.13+;
- `uv`;
- Docker Desktop / Docker Engine;
- Docker Compose v2.

## Установка и тесты

```powershell
uv lock --check
uv sync --frozen
uv run python -m compileall -q src tests monitoring/mock_exporter
uv run pytest -q
```

Конфигурация должна корректно разбираться как YAML, Grafana dashboard — как JSON, а provisioning-файлы Grafana — как YAML.

## Конфигурация

Пользовательская конфигурация хранится в:

```text
config/config.yaml
```

В Docker она доступна как `/app/config/config.yaml`. Parser и модели конфигурации находятся в `src/drift_guardian/config_handler/`.

Один экземпляр `Config`, созданный `OfflineWrapper`, используется как источник настроек для:

- feature metrics;
- prediction metrics;
- global thresholds;
- local feature overrides;
- stream thresholds;
- adversarial validation.

Realtime-контур не содержит второго parser конфигурации и отдельной реализации AV.

### Global и local thresholds

Глобальные значения задаются в `thresholds`. Локальный override задаётся в `features.<feature>.thresholds`.

В текущем примере:

```text
age.psi    → local:  warning=0.05, critical=0.12
income.psi → global: warning=0.10, critical=0.25
```

Core формирует `resolved_thresholds`; exporter публикует как глобальные, так и фактически применённые значения:

```text
drift_threshold{metric,level}
drift_resolved_threshold{feature,type,metric,level}
```

## Realtime windows и Kafka offsets

События собираются в полные непересекающиеся окна размером `WINDOW_SIZE`.

После успешного анализа полного окна выполняется последовательность:

1. `WindowBuffer` преобразуется в `pandas.DataFrame`.
2. Core выполняет schema check и drift-анализ.
3. Результат экспортируется в Prometheus.
4. Увеличивается счётчик завершённых анализов.
5. Kafka offsets подтверждаются синхронным commit.
6. Окно и window-local stream state сбрасываются.

Если анализ или экспорт завершается ошибкой, commit полного окна не выполняется.

## Stream health

Текущие stream-метрики:

```text
drift_event_time_lag_seconds
drift_window_time_span_seconds
drift_max_event_gap_seconds
drift_invalid_event_time_rate
drift_late_event_rate
```

Lifetime counters:

```text
drift_late_events_total
drift_out_of_order_events_total
```

`drift_stream_status` рассчитывается только по stream-метрикам, для которых в `stream_drift` заданы warning/critical thresholds. Метрики без thresholds остаются информационными и не влияют на общий status.

`drift_late_event_rate` является window-local метрикой. Lifetime counters не используются как текущий health signal и не переводят поток в permanent critical state.

## Adversarial validation

AV настраивается только через блок `adversarial_validation` в `config/config.yaml`:

```yaml
adversarial_validation:
  enabled: true
  interval_minutes: 30
  max_samples: 50000
  n_splits: 5
  random_state: 42
  missing_category: "__missing__"
```

Realtime вызывает Core API:

```text
OfflineWrapper.run_av(...)
  ↓
DriftMetricsEngine.run_adversarial_validation(...)
```

Monitoring получает ROC AUC и fold-level diagnostics:

```text
drift_av_roc_auc
drift_av_roc_auc_cv_mean
drift_av_roc_auc_cv_min
drift_av_roc_auc_cv_max
drift_av_roc_auc_cv_std
drift_av_driver_consistency
drift_av_driver_similarity_previous
```

`drift_av_driver_consistency` вычисляется внутри Core AV как средняя cosine similarity feature-importance между CV-фолдами одного запуска. Realtime-слой не пересчитывает эту метрику.

`drift_av_driver_similarity_previous` сравнивает feature importance текущего и предыдущего завершённого AV. Поэтому на первом AV после старта analyzer значение недоступно; со второго завершённого AV в рамках того же процесса exporter публикует similarity. Обычные analysis windows между AV-запусками не создают новое значение: при `interval_minutes: 30` второй similarity появляется только после второго планового AV, а не после второго 1000-event окна.

На окнах, где AV не запускается из-за `interval_minutes`, exporter сохраняет последний успешно опубликованный AV snapshot.

В верхней metadata-панели Grafana наличие prediction monitoring определяется по опубликованным `drift_threshold{metric="prediction_..."}`. Эти thresholds exporter публикует при старте, поэтому dashboard не показывает `Prediction: Not included` только из-за того, что первое analysis window ещё не завершено.

## Prometheus contract

Основные drift-метрики:

```text
drift_overall_status
drift_active_alerts
drift_status_feature{feature,type}
drift_metric_value{feature,type,metric}
drift_status{feature,type,metric}
drift_threshold{metric,level}
drift_resolved_threshold{feature,type,metric,level}
```

Realtime state:

```text
drift_window_size
drift_current_window_events
drift_events_processed_total
drift_analysis_runs_total
drift_last_analysis_age_seconds
drift_stream_status
```

Коды статусов:

```text
-1 = insufficient_data / not configured
 0 = ok
 1 = warning
 2 = critical
```

## Grafana и alerting

Grafana provisioning находится в:

```text
monitoring/grafana/provisioning/alerting/
  alert-rules.yaml
  contact-points.yaml
  mute-timings.yaml
  policies.yaml
  templates.yaml
```

Alert rules для drift/AV используют подтверждение состояния на четырёх последовательных analysis windows. Отдельный alert по `stream_status` не создаётся.

Telegram contact point использует локальную переменную окружения:

```env
TELEGRAM_BOT_TOKEN=<real_bot_token>
```

`.env` не должен попадать в Git. Без непустого `TELEGRAM_BOT_TOKEN` Grafana не сможет provision Telegram contact point.

## Docker profiles

Используются два взаимоисключающих режима:

- `mock` — проверка monitoring-контракта без Kafka/Core;
- `realtime` — полный Kafka → Core → Prometheus pipeline.

`kafka`, `prometheus` и `grafana` являются общей инфраструктурой. При переключении режима они **не удаляются и не пересоздаются**. Также сохраняются project network, `grafana-data` и Docker images.

`drift-mock-exporter` и `analyzer` используют host port `8000`, поэтому одновременно запускать `mock` и `realtime` нельзя.

### Первый запуск общей инфраструктуры

```powershell
docker compose up -d kafka prometheus grafana
```

Kafka остаётся поднятой и в mock, и в realtime. `kafka-init` при этом остаётся realtime one-shot сервисом: он создаёт `features-stream` перед запуском analyzer и завершается с кодом `0`.

Проверка:

```powershell
docker compose ps
curl.exe -s http://localhost:3000/api/health
```

### Mock

Запуск только mock exporter:

```powershell
docker compose --profile mock up -d --build drift-mock-exporter
```

Проверка:

```powershell
docker compose --profile mock ps -a
curl.exe -s http://localhost:8000/metrics | Select-String "drift_"
curl.exe -s http://localhost:8000/metrics | Select-String "drift_resolved_threshold"
```

Перед переходом в realtime удалить только профильный mock-контейнер:

```powershell
docker compose --profile mock stop drift-mock-exporter
docker compose --profile mock rm -f drift-mock-exporter
```

Порт `8000` после этого должен быть свободен:

```powershell
docker ps --filter "publish=8000" --format "table {{.Names}}\t{{.Ports}}"
```

### Realtime

Запуск профильных realtime-сервисов без пересоздания Kafka, Prometheus и Grafana:

```powershell
docker compose --profile realtime up -d --build kafka-init analyzer drift-producer
```

Проверка состояния:

```powershell
docker compose --profile realtime ps -a
```

Ожидаемое состояние:

```text
kafka           Up (healthy)
kafka-init      Exited (0)
analyzer        Up (healthy)
drift-producer  Up
prometheus      Up
Grafana         Up
```

Проверка analyzer и exporter:

```powershell
docker compose --profile realtime logs --tail=200 analyzer
curl.exe -s http://localhost:8000/metrics | Select-String "drift_"
```

Prometheus target:

```text
http://localhost:9090/targets
```

Grafana:

```text
http://localhost:3000
```

Перед возвратом в mock удалить только профильные realtime-контейнеры:

```powershell
docker compose --profile realtime stop drift-producer analyzer
docker compose --profile realtime rm -f drift-producer analyzer kafka-init
```

Kafka, Prometheus, Grafana, `drift_guardian_default`, `grafana-data` и images при обычном переключении сохраняются.


### Разделение истории mock и realtime

Prometheus хранит mock и realtime как отдельные scrape targets. Публичный label `job="drift-exporter"` сохраняется, а источники различаются стандартным Prometheus label `instance`. Дополнительный `mode` label не добавляется, поэтому техническое различие источников не появляется отдельной колонкой в таблицах Grafana.

History-панели Grafana фильтруются по exporter instance, активному в конце выбранного периода. Поэтому после переключения:

```text
mock     -> отображаются mock prediction metrics;
realtime -> отображаются только prediction metrics, реально включённые в config.
```

В текущем realtime config для prediction включён только `psi`, поэтому старые mock-линии JS divergence / KS test / Missing rate / Wasserstein не смешиваются с realtime-графиком. История Prometheus при этом не удаляется.

### Полный сброс Docker-состояния проекта

Полная очистка не используется для обычного переключения режимов. Она нужна только для явного сброса всего Compose project:

```powershell
docker compose --profile mock --profile realtime down --remove-orphans --volumes
```

При необходимости дополнительно удалить локально собранные project images:

```powershell
docker compose --profile mock --profile realtime down --remove-orphans --volumes --rmi local
```

## Диагностика realtime startup

Если `analyzer` долго остаётся в `Waiting` / `health: starting`, проверить:

```powershell
docker compose --profile realtime ps -a
docker compose --profile realtime logs --tail=200 analyzer
docker inspect drift_guardian-analyzer-1 --format "{{json .State.Health}}"
docker compose --profile realtime logs kafka-init
```

`kafka-init` должен завершиться с кодом `0`; topic `features-stream` должен быть доступен по `kafka:19092` внутри Compose network.

## Offline HTML report

```powershell
uv run drift-guardian-report <report.json> <output.html>
```

`generate_html_report()` принимает как in-memory mapping, так и путь к JSON-файлу.

## Код-стиль

- imports — через `drift_guardian...`;
- публичные функции и структуры данных имеют type hints;
- нетривиальные функции и методы имеют docstring;
- технические docstring и комментарии в Python-коде оформляются на русском языке;
- комментарии описывают контракт или причину решения и не дублируют код;
- realtime-слой не дублирует Core/config/AV реализацию.

## Секреты и локальные данные

Не коммитятся:

- `.env`;
- Telegram bot token;
- локальные datasets;
- сгенерированные отчёты;
- временные backup-файлы.
