# Data Drift System Realtime

Realtime-часть проекта Data Drift Guardian

Репозиторий реализует цепочку:

```text
Kafka Producer
    ↓
Kafka topic: features-stream
    ↓
Kafka Consumer
    ↓
Sliding Window
    ↓
DriftMetricsEngine
    ↓
PrometheusExporter
    ↓
/metrics
    ↓
Prometheus
```

В проект входят producer, consumer, count-based sliding window, адаптер к drift
engine, Prometheus exporter и Docker Compose

Grafana, Alertmanager, firing rules, batch-анализ и расчёт статистических
метрик внутри realtime-слоя не реализуются

До интеграции с Core Metrics используется `MockDriftEngine`. Он проверяет
realtime-инфраструктуру, но не заменяет настоящий расчёт drift

## Структура

```text
data-drift-system-realtime/
├── src/
│   └── drift_guardian/
│       ├── realtime/
│       │   ├── config.py
│       │   ├── consumer.py
│       │   ├── engine_adapter.py
│       │   ├── event.py
│       │   ├── producer.py
│       │   ├── realtime_monitor.py
│       │   └── window_buffer.py
│       └── exporters/
│           └── prometheus_exporter.py
├── prometheus/
│   └── prometheus.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Kafka event

Обязательные поля:

- `event_id`
- `event_time` в timezone-aware ISO-8601 формате

`prediction_score` опционален и обрабатывается отдельно от обычных features

Остальные плоские scalar-поля считаются признаками

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

Поддерживаются значения:

```text
int / float / str / bool / null
```

Вложенные `list` и `dict` отклоняются

## Sliding window

По умолчанию:

```text
WINDOW_SIZE=1000
MIN_WINDOW_SIZE=300
ANALYZE_EVERY_N_EVENTS=1
```

`drift_window_size` показывает текущее число событий в окне

```text
100 событий  -> drift_window_size = 100
300 событий  -> drift_window_size = 300
1000 событий -> drift_window_size = 1000
1200 событий -> drift_window_size = 1000
```

После заполнения окна новое событие вытесняет самое старое

До `MIN_WINDOW_SIZE` статус считается `insufficient_data`

## Prerequisites

При проверке и запуске обязательны:

- Docker Desktop на Windows/macOS или Docker Engine на Linux
- Docker Compose v2
- Python 3.12+ только для локальной проверки Python-кода

Проверить Docker:

```bash
docker --version
docker compose version
```

## Запуск

Все команды Docker ниже одинаковы для Windows, macOS и Linux

Из корневой папки проекта:

```bash
docker compose up --build -d
```

Проверить контейнеры:

```bash
docker compose ps
```

Ожидается:

```text
kafka            Up / healthy
kafka-init       Exited (0)
drift-consumer   Up / healthy
drift-producer   Up
prometheus       Up
```

`kafka-init` является одноразовым сервисом, поэтому `Exited (0)` для него
нормален

## Проверка Kafka

Проверить topic:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --list
```

Ожидается:

```text
features-stream
```

Посмотреть логи producer и consumer:

```bash
docker compose logs --tail=50 drift-producer
docker compose logs --tail=50 drift-consumer
```

Для просмотра логов в реальном времени:

```bash
docker compose logs -f drift-producer drift-consumer
```

## Проверка Prometheus exporter

Открыть:

```text
http://localhost:8000/metrics
```

Ключевые метрики:

```text
drift_overall_status
drift_active_alerts
drift_window_size
drift_events_processed_total

drift_feature_status{feature,type}
drift_feature_psi{feature,type}
drift_feature_missing_rate{feature,type}
drift_feature_mean_zscore{feature,type}
drift_feature_unseen_category_rate{feature,type}
drift_feature_cardinality_ratio{feature,type}

drift_prediction_status
drift_prediction_psi
drift_prediction_positive_rate

drift_stream_status
drift_event_time_lag_seconds
drift_window_time_span_seconds
drift_max_event_gap_seconds
drift_invalid_event_time_rate
drift_late_events_total
drift_out_of_order_events_total
```

Для demo-потока:

- `drift_events_processed_total` должен расти
- `drift_window_size` растёт до `1000`, затем остаётся равным `1000`
- `prediction_score` не должен появляться как обычная feature
- `drift_prediction_*` должны находиться в отдельном prediction-блоке
- `drift_late_events_total` и `drift_out_of_order_events_total` могут оставаться `0`
- `*_created` series отключены

## Проверка Prometheus

Открыть:

```text
http://localhost:9090/targets
```

Target `drift-consumer` должен иметь состояние `UP`

Prometheus UI:

```text
http://localhost:9090
```

Примеры запросов:

```promql
drift_events_processed_total
```

```promql
drift_window_size
```

```promql
drift_event_time_lag_seconds
```

```promql
drift_stream_status
```

## Проверка Python-кода

Создать виртуальное окружение:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверить стиль и синтаксис кода:

```bash
ruff check src
ruff format --check src
python -m compileall -q src
```

Проверить Docker Compose:

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

## Интеграция с Core Metrics

Realtime-слой ожидает движок с интерфейсом:

```python
def analyze_dataframe(current_df: pandas.DataFrame) -> dict:
    ...
```

Сейчас используется:

```python
MockDriftEngine()
```

После merge он заменяется командным `DriftMetricsEngine`

`event_id` и `event_time` не передаются в DataFrame как ML-фичи

`prediction_score` остаётся в DataFrame для prediction-анализатора Core, но в
Prometheus не дублируется как обычная feature

Пороги PSI, `mean_zscore`, `unseen_category_rate` и других Core Metrics должны
оставаться в Core-конфиге. Realtime-часть хранит только собственные пороги
качества потока

## Остановка

```bash
docker compose down --remove-orphans
```
