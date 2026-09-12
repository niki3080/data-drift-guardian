# Архитектура Data Drift Guardian 

## Основной принцип

```text
reference data → Profiler → reference profile
                                    │
current data ───────────────→ Drift Engine
                                    │
                                    ▼
                               drift report
                                │         │
                                ▼         ▼
                          HTML report  Prometheus
```

## Предполагаемая структура

```text
drift_guardian/
│
│   # КОД ПРОЕКТА
│
├── src/
│   └── drift_guardian/
│       ├── core/                    # Лев: profiler, config parser, metrics, engine
│       ├── realtime/                # Дима: события, Kafka, окна, consumer/producer
│       ├── exporters/               # Дима: экспорт drift report в Prometheus
│       └── reporting/               # Юля: генерация offline-отчётов
│
├── monitoring/                      # Юля: Grafana и интеграция с Prometheus
├── tests/                           
│
│   # ПОЛЬЗОВАТЕЛЬСКИЕ КОНФИГИ И АРТЕФАКТЫ
│
├── config/
│   └── config.yaml                  # Лев: пример пользовательской конфигурации
├── profiles/                        # ?? сгенерированные baseline-профили
└── reports/                         # сгенерированные offline-отчёты
│
│   # ДОКУМЕНТАЦИЯ И ПРИМЕРЫ
│
├── README.md
├── docs/
│   ├── architecture.md              # общая архитектура и границы блоков
│   ├── integration.md               # ?? общие форматы и контракты
│   ├── core.md                      # Лев
│   ├── realtime.md                  # Дима
│   └── monitoring.md                # Юля
└── notebooks/                       # Юля: демонстрационные ноутбуки
│
│   # СБОРКА И ЗАПУСК
│
├── Dockerfile                       # Дима ?? основа – Dockerfile realtime
├── docker-compose.yml               # будущий общий стек
├── docker-compose-yulia.yml         # временный mock-стек для Grafana
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── .dockerignore
```

### <strong>src/drift_guardian/core/</strong> — расчёт метрик (<i>Лев</i>)

Структуру своего блока определяет Лев:

```text
src/drift_guardian/
└──core/                         
    ├── config/parse_config.py              # хранит Pydantic-модели пользовательской drift-конфигурации     
    ├── profiling/baseline_profiler.py      # строит baseline по эталонному `DataFrame`
    ├── metrics/registry.py                 # Сопоставляет имя метрики из config с Python-функцией        
    ├── metrics/...                         # функции расчёта метрик
    └── .../engine.py                       # связывает config, reference profile и функции метрик
```

### <strong>src/drift_guardian/realtime/</strong> – получение событий и управление окном (<i>Дмитрий</i>)</summary>

Структуру своего блока определяет Дмитрий.  
Документация: [Realtime Pipeline](realtime.md)

```text
src/drift_guardian/
└── realtime/                             
    ├── config.py
    ├── consumer.py
    ├── engine_adapter.py
    ├── event.py
    ├── producer.py
    ├── realtime_monitor.py
    └── window_buffer.py
```

### <strong>src/drift_guardian/exporters/</strong> – отправка готового результата наружу (<i>Дмитрий</i>)</summary>

Структуру своего блока определяет Дмитрий:

```text
src/drift_guardian/
└── exporters/                          
    └── prometheus_exporter.py
```

### <strong>src/drift_guardian/reporting/</strong> — оффлайн-репорты (<i>Юлия</i>)</summary>

```text
src/drift_guardian/
└── reporting/                    
    ├── __init__.py               
    ├── offline_report.py          
    └── styles/
        └── report.css            
```

<hr>

### <strong>monitoring/</strong> — Grafana и интеграция с Prometeus (<i>Юлия</i>)</summary>

```text
 monitoring/                            
├── prometheus/
│   └── prometheus.yml                
├── grafana/
│   ├── dashboards/
│   │   └── drift_guardian.json       
│   └── provisioning/                 
│       ├── dashboards.yaml            
│       └── prometeus.yaml            
└── mock_exporter/                    # ВРЕМЕННО: для demo-разработки
```

