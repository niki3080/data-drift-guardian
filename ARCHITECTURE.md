# Архитектура Data Drift Guardian

`Data Drift Guardian` — система детекции дрейфа и контроля качества данных в пакетном и потоковом режимах.

## Верхнеуровневая структура

```text
drift_guardian/
├── metrics/       # считает метрики и выполняет статистические тесты
├── profiling/     # строит baseline, с которым сравниваются новые данные
├── batch/         # применяет метрики к двум датафреймам
├── realtime/      # применяет метрики к sliding window
├── dashboard/     # отображает результаты мониторинга
├── exporters/     # отправляет результаты во внешние системы
└── config/        # хранит описание признаков и пороговые значения
```

## Детализированная структура

<details>
<summary><strong>metrics/ — расчёт метрик</strong></summary>

```text
metrics/
├── __init__.py
├── psi.py
├── statistical_tests.py
├── distances.py
├── missingness.py
├── categorical.py
├── numeric.py
├── prediction.py       # prediction score drift
└── registry.py         # сопоставление metric_name → function/class
```

</details>

<details>
<summary><strong>profiling/ — построение baseline</strong></summary>

```text
profiling/
└── baseline_profiler.py
```

```text
reference.csv
-> BaselineProfiler
-> reference_profile.json
```

</details>

<details>
<summary><strong>batch/ — пакетный анализ</strong></summary>

```text
batch/
├── batch_analyzer.py
├── offline_report.py
└── adversarial_validation.py
```

</details>

<details>
<summary><strong>realtime/ — потоковый мониторинг</strong></summary>

```text
realtime/
├── producer.py
├── consumer.py
├── window_buffer.py
└── realtime_monitor.py
```

</details>

<details>
<summary><strong>dashboard/ — визуализация результатов</strong></summary>

```text
dashboard/
├── app.py
├── overview.py
├── data_quality.py
├── input_drift.py
├── feature_details.py
└── components.py
```

```text
OFFLINE MONITORING DASHBOARD (html)
│
├── Overview                    # верхнеуровневый сводный дашборд
│   ├── Dataset information     # контекст датасета
│   │   ├── Dataset name: <name>
│   │   ├── Reference rows: <num> / Current rows: <num>
│   │   └── Monitored features
│   │       ├── Numerical: <num> [feature list]
│   │       ├── Categorical: <num> [feature list]
│   │       ├── Datetime: <num> [feature list]
│   │       └── Ignored: <num> [feature list]
│   │
│   ├── Data Quality
│   │   ├── Schema check: <num> missing / <num> extra columns
│   │   ├── Dtype check: <num> columns with dtype mismatch
│   │   ├── Missing values check: <num> features affected
│   │   ├── Duplicate rows check: <num> duplicates
│   │   ├── Unseen categories check: <num> features affected
│   │   ├── Numeric out-of-range check: <num> features affected
│   │   ├── Datetime validity check: <num> features affected
│   │   └── Data quality alerts: [alert list]
│   │
│   ├── Data Drift
│   │   ├── Univariate drift
│   │   │   ├── Feature distribution drift: <num> features affected
│   │   │   ├── Missing values drift: <num> features affected
│   │   │   └── Maximum feature PSI: <value> (<feature>)
│   │   ├── Multivariate drift
│   │   │   └── Dataset separability (ROC-AUC): <value>
│   │   └── Data drift alerts: [alert list]
│   │
│   └── Feature Details
│       └── Table with Data Quality and Data Drift results by feature
```

```text
Feature Details
├── Feature                  # название признака
├── Type                     # numerical / categorical / datetime
├── Reference missing rate   # доля пропусков в baseline
├── Current missing rate     # доля пропусков в текущих данных
├── Missingness drift        # изменение доли пропусков
├── Drift metric             # PSI / KS / χ² / Wasserstein / JS
├── Metric value             # значение drift-метрики
├── P-value                  # значимость KS или χ²; иначе N/A
├── Threshold                # порог срабатывания алерта
├── Data quality status      # статус качества признака [passed / warning / error]
├── Drift status             # статус дрейфа признака  [passed / warning / error]
└── Alert message            # краткое описание проблемы
```
</details>

<details>
<summary><strong>exporters/ — отправка результатов</strong></summary>

```text
exporters/
└── prometheus_exporter.py
```

</details>

<details>
<summary><strong>config/ — конфигурация</strong></summary>

```text
config/
├── features.yaml
└── thresholds.yaml
```

</details>
