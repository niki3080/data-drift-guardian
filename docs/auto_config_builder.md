## Назначение

`auto_config_builder.py` нужен для автоматической генерации YAML-конфига мониторинга drift'а по reference dataset.

На вход подаётся `pandas.DataFrame`, который считается **reference data** / baseline data.

На выходе получается:

- Python-словарь с конфигом;
- опционально YAML-файл, если передан `output_path`.

Генератор умеет:

- автоматически определять типы колонок: `numeric` / `categorical`;
- назначать дефолтные метрики по типам фичей;
- добавлять глобальные thresholds;
- добавлять thresholds для конкретной фичи;
- отключать фичи;
- отключать метрики глобально или точечно;
- переопределять список метрик для конкретной фичи;
- автоматически подбирать thresholds, если они не заданы вручную;
- учитывать направление threshold'ов для обычных и reversed-метрик;
- отдельно настраивать `prediction_metrics`;
- генерировать конфиг, совместимый со схемой `drift_guardian`.

---

## Важные импорты внутри генератора

Генератор использует registry метрик:

```python
from src.drift_guardian.analyzer.regestry.metric_registry import METRIC_REGISTRY
```

И profiler:

```python
from src.drift_guardian.profiler.baseline_profiler import Profiler
```

Также генератор использует список reversed-метрик из парсера конфига:

```python
from config.parse_config import REVERSED_THRESHOLD_METRICS
```

Все функции расчёта метрик берутся из `METRIC_REGISTRY`.

Важно:

```text
Metric.kstest — это D-статистика, а не p_value.
```

То есть threshold для `kstest` должен интерпретироваться как порог на значение D-статистики: чем больше значение, тем сильнее drift.

Важно:

```text
Metric.chi2 — это p_value.
```

Для `chi2` направление threshold'ов обратное: чем меньше p_value, тем сильнее drift.

---

## Быстрый старт

Минимальный вызов:

```python
from config.auto_config_builder import build_drift_config

config = build_drift_config(
    df,
    output_path="config.yaml",
)
```

Если файл сохранять не нужно:

```python
from config.auto_config_builder import build_drift_config

config = build_drift_config(df)
```

В этом режиме генератор сам:

1. берёт все колонки из `df`;
2. исключает datetime / timedelta колонки;
3. исключает полностью пустые колонки;
4. автоматически определяет тип каждой фичи;
5. назначает дефолтные метрики;
6. строит baseline profile через `Profiler`;
7. автоматически подбирает thresholds;
8. применяет дефолтные direction-aware ограничители для thresholds;
9. возвращает `dict`;
10. если передан `output_path`, сохраняет YAML.

---

## Рекомендуемый минимальный вызов

На практике лучше сразу явно исключить технические колонки и указать размер окна:

```python
from config.auto_config_builder import (
    build_drift_config,
    ConfigBuildOptions,
    AutoThresholdSettings,
)

config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        exclude_columns=["id", "created_at"],

        profiler_window_size=1000,

        auto_thresholds=AutoThresholdSettings(
            enabled=True,
            n_windows=200,
            warning_quantile=0.95,
            critical_quantile=0.99,
            random_state=42,
        ),
    ),
    output_path="config.yaml",
)
```

---

## Основная функция

```python
build_drift_config(
    df: pd.DataFrame,
    options: ConfigBuildOptions | None = None,
    output_path: str | None = None,
) -> dict[str, Any]
```

Параметры:

| Параметр | Тип | Описание |
|---|---|---|
| `df` | `pd.DataFrame` | Reference dataset, по которому строится baseline и оцениваются thresholds |
| `options` | `ConfigBuildOptions \| None` | Настройки генерации |
| `output_path` | `str \| None` | Путь для сохранения YAML-конфига |

Возвращает:

```python
dict[str, Any]
```

То есть готовый конфиг в виде Python-словаря.

---

## Общая структура результата

Генератор создаёт конфиг примерно такого вида:

```yaml
features:
  age:
    type: numeric
    metrics:
      - missing_rate
      - psi
      - kstest

  country:
    type: categorical
    metrics:
      - missing_rate
      - psi
      - unseen_category_rate
      - chi2

prediction_metrics:
  enabled: false

thresholds:
  missing_rate:
    warning: 0.005
    critical: 0.02
  psi:
    warning: 0.1
    critical: 0.25
  kstest:
    warning: 0.02
    critical: 0.05
  unseen_category_rate:
    warning: 0.005
    critical: 0.02
  chi2:
    warning: 0.05
    critical: 0.01
```

Обратите внимание на `chi2`:

```yaml
chi2:
  warning: 0.05
  critical: 0.01
```

Это корректно, потому что `chi2` возвращает `p_value`, а для p_value меньшее значение означает более сильный drift.

Если для фичи заданы локальные thresholds, они попадут внутрь блока конкретной фичи:

```yaml
features:
  age:
    type: numeric
    metrics:
      - missing_rate
      - psi
      - kstest
    thresholds:
      psi:
        warning: 0.05
        critical: 0.12
```

---

## Приоритет thresholds

Threshold для метрики выбирается по следующему принципу:

1. если для фичи задан локальный threshold — используется он;
2. иначе используется глобальный threshold из блока `thresholds`;
3. если threshold не найден нигде — генератор пытается подобрать его автоматически;
4. если автоподбор выключен и threshold не задан — генерация завершится ошибкой.

Пример:

```python
ConfigBuildOptions(
    global_thresholds={
        "psi": {"warning": 0.1, "critical": 0.25},
    },
    feature_thresholds={
        "age": {
            "psi": {"warning": 0.05, "critical": 0.12},
        }
    },
)
```

В этом случае:

- для `age.psi` будет использоваться `0.05 / 0.12`;
- для остальных фичей с метрикой `psi` будет использоваться `0.1 / 0.25`.

---

## Направление thresholds

В Drift Guardian есть два типа метрик по направлению ухудшения.

### Обычные метрики

Для обычных метрик большее значение означает больший drift.

Для них должно выполняться:

```text
warning < critical
```

Примеры:

```yaml
psi:
  warning: 0.1
  critical: 0.25

missing_rate:
  warning: 0.005
  critical: 0.02

kstest:
  warning: 0.02
  critical: 0.05
```

Логика интерпретации:

```text
value >= warning  -> warning
value >= critical -> critical
```

### Reversed-метрики

Для reversed-метрик меньшее значение означает больший drift.

Для них должно выполняться:

```text
warning > critical
```

Пример:

```yaml
chi2:
  warning: 0.05
  critical: 0.01
```

`chi2` возвращает p_value, поэтому логика такая:

```text
p_value <= 0.05 -> warning
p_value <= 0.01 -> critical
```

Генератор берёт список reversed-метрик из:

```python
REVERSED_THRESHOLD_METRICS
```

То есть направление threshold'ов синхронизировано с парсером конфига.

---

## Определение типов фичей

По умолчанию генератор сам определяет типы колонок:

| Тип колонки в pandas | Тип в конфиге |
|---|---|
| numeric, кроме bool | `numeric` |
| object / string / category / bool | `categorical` |
| datetime / timedelta | исключается по умолчанию |

Пример автоматического определения:

```python
ConfigBuildOptions()
```

Если нужно задать типы явно:

```python
ConfigBuildOptions(
    feature_types={
        "age": "numeric",
        "income": "numeric",
        "country": "categorical",
        "is_premium": "categorical",
    }
)
```

---

## Исключение колонок

Чтобы исключить технические или ненужные колонки:

```python
ConfigBuildOptions(
    exclude_columns=["id", "created_at", "request_id"]
)
```

`exclude_columns` полностью убирает эти колонки из мониторинга как обычные фичи.

---

## Явный список колонок

Если нужно мониторить только конкретный набор колонок:

```python
ConfigBuildOptions(
    include_columns=["age", "income", "country"]
)
```

Если какая-то колонка из `include_columns` отсутствует в `df`, генератор выбросит ошибку.

---

## Отключение фичей

Есть отдельная настройка `disabled_features`:

```python
ConfigBuildOptions(
    disabled_features=["debug_column", "temporary_flag"]
)
```

Функционально это похоже на `exclude_columns`.

Можно использовать так:

- `exclude_columns` — для технических колонок;
- `disabled_features` — для бизнесового временного отключения мониторинга.

---

## Дефолтные метрики

По умолчанию для numeric-фичей используются:

```python
[
    "missing_rate",
    "psi",
    "js_divergence",
    "wasserstein_distance",
    "kstest",
]
```

Для categorical-фичей:

```python
[
    "missing_rate",
    "psi",
    "js_divergence",
    "unseen_category_rate",
    "cardinality_ratio",
    "chi2",
    "cramer_v",
    "category_churn",
]
```

Важно:

```text
kstest — это D-статистика, не p_value.
chi2 — это p_value.
```

---

## Переопределение глобальных наборов метрик

Можно задать свои наборы метрик для всех numeric и categorical фичей.

Пример:

```python
ConfigBuildOptions(
    numeric_metrics=[
        "missing_rate",
        "psi",
        "kstest",
    ],
    categorical_metrics=[
        "missing_rate",
        "psi",
        "unseen_category_rate",
        "chi2",
    ],
)
```

Тогда:

- все numeric-фичи получат `missing_rate`, `psi`, `kstest`;
- все categorical-фичи получат `missing_rate`, `psi`, `unseen_category_rate`, `chi2`.

---

## Отключение метрик глобально

Если нужно полностью отключить метрику для всех фичей:

```python
ConfigBuildOptions(
    disabled_metrics=["kstest", "chi2"]
)
```

Например, если `kstest` отключён глобально, он не будет добавлен ни для обычных фичей, ни для prediction metrics.

---

## Отключение метрик для конкретной фичи

Если нужно отключить метрику только для одной фичи:

```python
ConfigBuildOptions(
    feature_disabled_metrics={
        "age": ["wasserstein_distance"],
        "country": ["chi2", "cramer_v"],
    }
)
```

В этом примере:

- для `age` будет отключён `wasserstein_distance`;
- для `country` будут отключены `chi2` и `cramer_v`;
- остальные фичи продолжат использовать эти метрики, если они есть в их списке.

---

## Полное переопределение метрик для конкретной фичи

Если нужно задать индивидуальный список метрик для фичи:

```python
ConfigBuildOptions(
    feature_metrics={
        "income": ["missing_rate", "psi", "kstest"],
        "country": ["missing_rate", "unseen_category_rate", "chi2"],
    }
)
```

Важно:

```text
feature_metrics полностью заменяет дефолтный список метрик для указанной фичи.
```

То есть если для `income` указано:

```python
"income": ["missing_rate", "psi"]
```

то `kstest`, `wasserstein_distance` и другие numeric-метрики автоматически добавлены не будут.

---

## Совместимость метрик с типами фичей

Некоторые метрики имеют смысл только для numeric-фичей:

```python
[
    "wasserstein_distance",
    "kstest",
]
```

Некоторые — только для categorical-фичей:

```python
[
    "unseen_category_rate",
    "cardinality_ratio",
    "chi2",
    "cramer_v",
    "category_churn",
]
```

Если попытаться добавить несовместимую метрику, поведение зависит от настройки `strict_metric_compatibility`.

По умолчанию:

```python
strict_metric_compatibility=True
```

Это значит, что генератор выбросит ошибку.

Если поставить:

```python
ConfigBuildOptions(
    strict_metric_compatibility=False
)
```

то несовместимые метрики будут отброшены с warning'ом.

---

## Проверка наличия метрик в METRIC_REGISTRY

Генератор проверяет, что каждая метрика есть в `METRIC_REGISTRY`.

По умолчанию:

```python
strict_metric_registry=True
```

Если метрики нет в registry, генератор выбросит ошибку.

Можно изменить поведение:

```python
ConfigBuildOptions(
    strict_metric_registry=False
)
```

В этом случае неизвестные метрики будут отброшены с warning'ом.

---

## Ручная настройка глобальных thresholds

Глобальные thresholds задаются через `global_thresholds`.

Пример для обычных метрик:

```python
ConfigBuildOptions(
    global_thresholds={
        "missing_rate": {"warning": 0.005, "critical": 0.02},
        "psi": {"warning": 0.1, "critical": 0.25},
        "kstest": {"warning": 0.02, "critical": 0.05},
    }
)
```

Пример для reversed-метрики `chi2`:

```python
ConfigBuildOptions(
    global_thresholds={
        "chi2": {"warning": 0.05, "critical": 0.01},
    }
)
```

Можно использовать словарь:

```python
"psi": {"warning": 0.1, "critical": 0.25}
```

Или короткую форму tuple/list:

```python
"psi": (0.1, 0.25)
```

То есть эти две записи эквивалентны:

```python
global_thresholds={
    "psi": {"warning": 0.1, "critical": 0.25},
}
```

```python
global_thresholds={
    "psi": (0.1, 0.25),
}
```

Для reversed-метрик tuple/list тоже поддерживается:

```python
global_thresholds={
    "chi2": (0.05, 0.01),
}
```

---

## Ручная настройка thresholds для конкретной фичи

Локальные thresholds задаются через `feature_thresholds`.

Пример:

```python
ConfigBuildOptions(
    feature_thresholds={
        "age": {
            "psi": {"warning": 0.05, "critical": 0.12},
            "kstest": {"warning": 0.03, "critical": 0.08},
        },
        "country": {
            "chi2": {"warning": 0.05, "critical": 0.01},
        },
    }
)
```

В YAML это будет выглядеть примерно так:

```yaml
features:
  age:
    type: numeric
    metrics:
      - missing_rate
      - psi
      - kstest
    thresholds:
      psi:
        warning: 0.05
        critical: 0.12
      kstest:
        warning: 0.03
        critical: 0.08

  country:
    type: categorical
    metrics:
      - missing_rate
      - psi
      - chi2
    thresholds:
      chi2:
        warning: 0.05
        critical: 0.01
```

---

## Требования к threshold values

Для каждого threshold должны быть указаны два уровня:

```yaml
warning: ...
critical: ...
```

Общие правила:

1. оба значения должны быть неотрицательными;
2. значения должны быть числами;
3. направление зависит от типа метрики.

Для обычных метрик должно выполняться:

```text
warning < critical
```

Корректно:

```python
"psi": {"warning": 0.1, "critical": 0.25}
```

Некорректно:

```python
"psi": {"warning": 0.25, "critical": 0.1}
```

Для reversed-метрик должно выполняться:

```text
warning > critical
```

Корректно:

```python
"chi2": {"warning": 0.05, "critical": 0.01}
```

Некорректно:

```python
"chi2": {"warning": 0.01, "critical": 0.05}
```

Некорректно для любых метрик:

```python
"psi": {"warning": -0.1, "critical": 0.25}
```

---

## Автоматический подбор thresholds

Если threshold для используемой метрики не задан вручную, генератор может подобрать его автоматически.

За это отвечает:

```python
AutoThresholdSettings
```

По умолчанию автоподбор включён:

```python
AutoThresholdSettings(
    enabled=True
)
```

Общая идея:

1. строится baseline profile по reference `df`;
2. из этого же `df` генерируются calibration windows;
3. для каждого окна считаются drift-метрики против baseline;
4. получается распределение значений метрики на нормальных данных;
5. для обычных метрик используются верхние квантили;
6. для reversed-метрик используются нижние квантили;
7. применяются direction-aware ограничители из `floors`;
8. итоговые thresholds добавляются в конфиг.

Например:

```python
AutoThresholdSettings(
    enabled=True,
    warning_quantile=0.95,
    critical_quantile=0.99,
)
```

Для обычных метрик это значит:

- `warning` будет примерно 95%-квантилем;
- `critical` будет примерно 99%-квантилем.

Для reversed-метрик это значит:

- `warning` будет примерно 5%-квантилем;
- `critical` будет примерно 1%-квантилем.

То есть для `chi2`, который возвращает p_value:

```text
warning_quantile = 1 - 0.95 = 0.05
critical_quantile = 1 - 0.99 = 0.01
```

---

## Настройки AutoThresholdSettings

```python
AutoThresholdSettings(
    enabled=True,
    method="bootstrap",
    window_size=None,
    n_windows=100,
    warning_quantile=0.95,
    critical_quantile=0.99,
    per_feature=False,
    per_prediction=False,
    floors=<default_auto_threshold_floors>,
    eps=1e-12,
    random_state=42,
    ignore_metric_errors=True,
)
```

Поля:

| Поле | Тип | Значение по умолчанию | Описание |
|---|---|---:|---|
| `enabled` | `bool` | `True` | Включить или выключить автоподбор thresholds |
| `method` | `"bootstrap" \| "rolling"` | `"bootstrap"` | Метод генерации calibration windows |
| `window_size` | `int \| None` | `None` | Размер calibration window. Если `None`, используется `profiler_window_size` |
| `n_windows` | `int` | `100` | Количество calibration windows |
| `warning_quantile` | `float` | `0.95` | Квантиль для warning threshold |
| `critical_quantile` | `float` | `0.99` | Квантиль для critical threshold |
| `per_feature` | `bool` | `False` | Добавлять ли auto thresholds на уровне каждой фичи |
| `per_prediction` | `bool` | `False` | Добавлять ли auto thresholds в блок `prediction_metrics` |
| `floors` | `dict` | default factory | Direction-aware ограничители thresholds по метрикам |
| `eps` | `float` | `1e-12` | Минимальный зазор, если warning и critical совпали |
| `random_state` | `int \| None` | `42` | Seed для воспроизводимости |
| `ignore_metric_errors` | `bool` | `True` | Игнорировать ошибки расчёта отдельных метрик на calibration windows |

---

## Дефолтные floors / ограничители thresholds

По умолчанию генератор использует следующие direction-aware ограничители:

```python
{
    "missing_rate": {
        "warning": 0.005,  # 0.5%
        "critical": 0.02,  # 2%
    },
    "unseen_category_rate": {
        "warning": 0.005,  # 0.5%
        "critical": 0.02,  # 2%
    },
    "category_churn": {
        "warning": 0.005,  # 0.5%
        "critical": 0.02,  # 2%
    },
    "psi": {
        "warning": 0.1,
        "critical": 0.25,
    },
    "js_divergence": {
        "warning": 0.005,
        "critical": 0.02,
    },
    "kstest": {
        "warning": 0.02,
        "critical": 0.05,
    },
    "chi2": {
        "warning": 0.05,
        "critical": 0.01,
    },
    "cramer_v": {
        "warning": 0.02,
        "critical": 0.05,
    },
    "cardinality_ratio": {
        "warning": 0.02,
        "critical": 0.05,
    },
}
```

Эти значения нужны, чтобы на стабильном reference dataset не получать бессмысленные thresholds вроде:

```yaml
missing_rate:
  warning: 0.0
  critical: 1.0e-12
```

Теперь для `missing_rate` при полностью чистом reference dataset будет использоваться минимум:

```yaml
missing_rate:
  warning: 0.005
  critical: 0.02
```

Для `psi` используются стандартные практические пороги:

```yaml
psi:
  warning: 0.1
  critical: 0.25
```

Для `chi2` используются p_value-пороги:

```yaml
chi2:
  warning: 0.05
  critical: 0.01
```

---

## Как работают floors для обычных и reversed-метрик

Для обычных метрик `floors` работают как минимальные значения.

Например:

```python
AutoThresholdSettings(
    floors={
        "missing_rate": {"warning": 0.005, "critical": 0.02},
    }
)
```

Если автоподбор дал:

```yaml
missing_rate:
  warning: 0.0
  critical: 1.0e-12
```

то итог будет:

```yaml
missing_rate:
  warning: 0.005
  critical: 0.02
```

Для reversed-метрик `floors` работают как direction-aware cap.

Например, `chi2` возвращает p_value. Если bootstrap на нормальных данных дал слишком высокие p_value thresholds:

```yaml
chi2:
  warning: 0.99
  critical: 0.97
```

такие thresholds были бы слишком чувствительными: warning срабатывал бы почти всегда при `p_value <= 0.99`.

Поэтому для reversed-метрик ограничитель применяется в обратную сторону, и итог будет:

```yaml
chi2:
  warning: 0.05
  critical: 0.01
```

---

## Метод bootstrap

При `method="bootstrap"` генератор создаёт окна с возвращением.

Пример:

```python
AutoThresholdSettings(
    method="bootstrap",
    n_windows=200,
    window_size=1000,
)
```

Это значит:

- будет сгенерировано 200 окон;
- каждое окно будет размером 1000 строк;
- строки будут sampled из reference `df` с возвращением.

Это хороший дефолтный вариант для большинства задач.

---

## Метод rolling

При `method="rolling"` генератор берёт последовательные окна из `df`.

Пример:

```python
AutoThresholdSettings(
    method="rolling",
    n_windows=50,
    window_size=1000,
)
```

Такой режим полезен, если порядок строк в `df` имеет смысл, например данные отсортированы по времени.

---

## Ручное переопределение floors

Если нужно заменить дефолтные ограничители, передайте свой `floors`.

Пример:

```python
AutoThresholdSettings(
    floors={
        "missing_rate": {
            "warning": 0.01,
            "critical": 0.03,
        },
        "psi": {
            "warning": 0.1,
            "critical": 0.25,
        },
        "kstest": {
            "warning": 0.02,
            "critical": 0.05,
        },
        "chi2": {
            "warning": 0.05,
            "critical": 0.01,
        },
    }
)
```

Можно использовать короткую tuple-форму:

```python
AutoThresholdSettings(
    floors={
        "psi": (0.1, 0.25),
        "kstest": (0.02, 0.05),
        "chi2": (0.05, 0.01),
    }
)
```

Важно:

```text
Для обычных метрик warning < critical.
Для reversed-метрик warning > critical.
```

---

## Глобальные auto thresholds и feature-level auto thresholds

По умолчанию:

```python
per_feature=False
```

Это значит, что генератор добавляет auto thresholds в глобальный блок:

```yaml
thresholds:
  psi:
    warning: ...
    critical: ...
```

Если включить:

```python
AutoThresholdSettings(
    per_feature=True
)
```

то генератор также добавит локальные auto thresholds на каждую фичу:

```yaml
features:
  age:
    type: numeric
    metrics:
      - psi
      - kstest
    thresholds:
      psi:
        warning: ...
        critical: ...
      kstest:
        warning: ...
        critical: ...
```

Обычно рекомендуется начинать с:

```python
per_feature=False
```

А включать `per_feature=True`, если у фичей сильно разные масштабы и глобальный threshold получается слишком грубым.

---

## Автоподбор и ручные thresholds

Ручные thresholds имеют приоритет.

Если задано:

```python
ConfigBuildOptions(
    global_thresholds={
        "psi": {"warning": 0.1, "critical": 0.25},
    },
    auto_thresholds=AutoThresholdSettings(
        enabled=True,
    ),
)
```

то `psi` не будет перезаписан автоподбором.

Если для какой-то другой метрики threshold не задан, например для `kstest`, генератор рассчитает его автоматически.

---

## Отключение автоподбора thresholds

Если нужно полностью отключить автоподбор:

```python
ConfigBuildOptions(
    auto_thresholds=AutoThresholdSettings(
        enabled=False,
    ),
    global_thresholds={
        "missing_rate": {"warning": 0.005, "critical": 0.02},
        "psi": {"warning": 0.1, "critical": 0.25},
        "kstest": {"warning": 0.02, "critical": 0.05},
        "chi2": {"warning": 0.05, "critical": 0.01},
    },
)
```

Важно:

```text
Если автоподбор отключён, необходимо вручную задать thresholds для всех используемых метрик.
```

---

## Настройка prediction metrics

Чтобы мониторить prediction score, нужно включить `prediction_enabled` и указать колонку:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
)
```

Если `prediction_score_column` указан, эта колонка по умолчанию не будет дублироваться как обычная feature.

---

## Prediction metrics с явным типом

Тип prediction score можно определить автоматически, но лучше указать явно.

Для числового score:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
    prediction_type="numeric",
)
```

Для категориального prediction:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="predicted_class",
    prediction_type="categorical",
)
```

---

## Дефолтные prediction metrics

Если `prediction_type="numeric"`, по умолчанию используются:

```python
["psi", "kstest"]
```

Если `prediction_type="categorical"`, по умолчанию используется:

```python
["psi"]
```

Можно задать список вручную:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
    prediction_type="numeric",
    prediction_metrics=["psi", "kstest", "wasserstein_distance"],
)
```

---

## Thresholds для prediction metrics

Prediction metrics могут использовать глобальные thresholds:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
    global_thresholds={
        "psi": {"warning": 0.1, "critical": 0.25},
        "kstest": {"warning": 0.02, "critical": 0.05},
    },
)
```

Или локальные thresholds внутри `prediction_metrics`:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
    prediction_thresholds={
        "psi": {"warning": 0.05, "critical": 0.15},
        "kstest": {"warning": 0.03, "critical": 0.08},
    },
)
```

---

## Автоподбор thresholds для prediction metrics

По умолчанию auto thresholds для prediction block добавляются через глобальные thresholds.

Если нужно добавить именно локальные thresholds внутрь блока `prediction_metrics`, включите:

```python
AutoThresholdSettings(
    per_prediction=True
)
```

Пример:

```python
ConfigBuildOptions(
    prediction_enabled=True,
    prediction_score_column="prediction_score",
    auto_thresholds=AutoThresholdSettings(
        enabled=True,
        per_prediction=True,
    ),
)
```

---

## Настройки Profiler

Генератор создаёт `Profiler` примерно так:

```python
Profiler(
    ref_data=df,
    window_size=options.profiler_window_size,
    num_features=num_features,
    cat_features=cat_features,
    prediction=prediction_col,
    merge_threshold=options.merge_threshold,
    low_cardinality_threshold=options.low_cardinality_threshold,
    take_sample=options.profiler_take_sample,
    sample_float_dtype=options.sample_float_dtype,
    random_state=options.random_state,
)
```

Настройки:

| Поле | Значение по умолчанию | Описание |
|---|---:|---|
| `profiler_window_size` | `1000` | Размер окна profiler'а |
| `merge_threshold` | `5` | Порог объединения редких категорий |
| `low_cardinality_threshold` | `15` | Порог низкой кардинальности |
| `profiler_take_sample` | `True` | Брать ли sample в profiler |
| `sample_float_dtype` | `"float32"` | dtype для sample float values |
| `random_state` | `42` | Seed для воспроизводимости |

Пример:

```python
ConfigBuildOptions(
    profiler_window_size=2000,
    merge_threshold=10,
    low_cardinality_threshold=20,
    profiler_take_sample=True,
    sample_float_dtype="float32",
    random_state=42,
)
```

---

## metric_kwargs

Если отдельные metric functions принимают дополнительные параметры, их можно передать через `metric_kwargs`.

Пример:

```python
ConfigBuildOptions(
    metric_kwargs={
        "psi": {
            "eps": 1e-6,
        },
        "js_divergence": {
            "base": 2,
        },
    }
)
```

Эти kwargs будут переданы в metric function при автоподборе thresholds:

```python
fn(reference_dict, current, **kwargs)
```

---

## stream_drift

Если в конфиге нужен блок `stream_drift`, его можно передать напрямую:

```python
ConfigBuildOptions(
    stream_drift={
        "drift_event_time_lag_seconds": {
            "warning": 30,
            "critical": 120,
        },
        "drift_late_events_total": {
            "warning": 10,
            "critical": 50,
        },
    }
)
```

Или в короткой форме:

```python
ConfigBuildOptions(
    stream_drift={
        "drift_event_time_lag_seconds": (30, 120),
        "drift_late_events_total": (10, 50),
    }
)
```

В YAML:

```yaml
stream_drift:
  drift_event_time_lag_seconds:
    warning: 30.0
    critical: 120.0
  drift_late_events_total:
    warning: 10.0
    critical: 50.0
```

Автоматический подбор thresholds для `stream_drift` не выполняется. Эти пороги нужно задавать явно.

Важно:

```text
stream_drift thresholds всегда используют обычное направление: warning < critical.
REVERSED_THRESHOLD_METRICS применяются только к метрикам drift'а.
```

---

## Полный пример настройки

```python
from config.auto_config_builder import (
    build_drift_config,
    ConfigBuildOptions,
    AutoThresholdSettings,
)

config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        include_columns=[
            "age",
            "income",
            "country",
            "device_type",
            "prediction_score",
        ],

        exclude_columns=[
            "id",
            "created_at",
        ],

        feature_types={
            "age": "numeric",
            "income": "numeric",
            "country": "categorical",
            "device_type": "categorical",
        },

        numeric_metrics=[
            "missing_rate",
            "psi",
            "kstest",
            "wasserstein_distance",
        ],

        categorical_metrics=[
            "missing_rate",
            "psi",
            "unseen_category_rate",
            "cardinality_ratio",
            "chi2",
        ],

        feature_metrics={
            "income": [
                "missing_rate",
                "psi",
                "kstest",
            ],
        },

        feature_disabled_metrics={
            "age": [
                "wasserstein_distance",
            ],
            "country": [
                "cardinality_ratio",
            ],
        },

        global_thresholds={
            "missing_rate": {
                "warning": 0.005,
                "critical": 0.02,
            },
            "psi": {
                "warning": 0.1,
                "critical": 0.25,
            },
            "chi2": {
                "warning": 0.05,
                "critical": 0.01,
            },
        },

        feature_thresholds={
            "age": {
                "psi": {
                    "warning": 0.05,
                    "critical": 0.12,
                },
                "kstest": {
                    "warning": 0.03,
                    "critical": 0.08,
                },
            },
            "country": {
                "chi2": {
                    "warning": 0.05,
                    "critical": 0.01,
                },
            },
        },

        prediction_enabled=True,
        prediction_score_column="prediction_score",
        prediction_type="numeric",
        prediction_metrics=[
            "psi",
            "kstest",
        ],
        prediction_thresholds={
            "psi": {
                "warning": 0.05,
                "critical": 0.15,
            },
        },

        stream_drift={
            "drift_event_time_lag_seconds": (30, 120),
            "drift_late_events_total": (10, 50),
        },

        profiler_window_size=1000,
        merge_threshold=5,
        low_cardinality_threshold=15,
        profiler_take_sample=True,
        sample_float_dtype="float32",
        random_state=42,

        auto_thresholds=AutoThresholdSettings(
            enabled=True,
            method="bootstrap",
            window_size=1000,
            n_windows=200,
            warning_quantile=0.95,
            critical_quantile=0.99,
            per_feature=False,
            per_prediction=False,
            floors={
                "missing_rate": {
                    "warning": 0.005,
                    "critical": 0.02,
                },
                "psi": {
                    "warning": 0.1,
                    "critical": 0.25,
                },
                "kstest": {
                    "warning": 0.02,
                    "critical": 0.05,
                },
                "chi2": {
                    "warning": 0.05,
                    "critical": 0.01,
                },
            },
            random_state=42,
        ),
    ),
    output_path="config.yaml",
)
```

---

## Пример с максимально простым конфигом

```python
from config.auto_config_builder import build_drift_config

config = build_drift_config(
    df,
    output_path="config.yaml",
)
```

Подходит для первого запуска и быстрой проверки.

---

## Пример только с ручными thresholds

```python
from config.auto_config_builder import (
    build_drift_config,
    ConfigBuildOptions,
    AutoThresholdSettings,
)

config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        numeric_metrics=[
            "missing_rate",
            "psi",
            "kstest",
        ],
        categorical_metrics=[
            "missing_rate",
            "psi",
            "unseen_category_rate",
            "chi2",
        ],
        global_thresholds={
            "missing_rate": {
                "warning": 0.005,
                "critical": 0.02,
            },
            "psi": {
                "warning": 0.1,
                "critical": 0.25,
            },
            "kstest": {
                "warning": 0.02,
                "critical": 0.05,
            },
            "unseen_category_rate": {
                "warning": 0.005,
                "critical": 0.02,
            },
            "chi2": {
                "warning": 0.05,
                "critical": 0.01,
            },
        },
        auto_thresholds=AutoThresholdSettings(
            enabled=False,
        ),
    ),
    output_path="config.yaml",
)
```

---

## Пример с feature-level auto thresholds

```python
from config.auto_config_builder import (
    build_drift_config,
    ConfigBuildOptions,
    AutoThresholdSettings,
)

config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        auto_thresholds=AutoThresholdSettings(
            enabled=True,
            per_feature=True,
            n_windows=200,
            warning_quantile=0.95,
            critical_quantile=0.99,
        ),
    ),
    output_path="config.yaml",
)
```

В этом режиме thresholds будут добавлены внутрь каждой фичи.

---

## Пример отключения дорогих метрик

Если некоторые метрики считаются слишком долго, их можно отключить:

```python
ConfigBuildOptions(
    disabled_metrics=[
        "chi2",
        "cramer_v",
        "wasserstein_distance",
    ]
)
```

Или отключить только для отдельных фичей:

```python
ConfigBuildOptions(
    feature_disabled_metrics={
        "country": [
            "chi2",
            "cramer_v",
        ],
        "income": [
            "wasserstein_distance",
        ],
    }
)
```

---

## Пример настройки только selected features

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        include_columns=[
            "age",
            "income",
            "country",
        ],
        feature_types={
            "age": "numeric",
            "income": "numeric",
            "country": "categorical",
        },
    ),
    output_path="config.yaml",
)
```

---

## Пример настройки prediction score

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        exclude_columns=[
            "id",
            "created_at",
        ],
        prediction_enabled=True,
        prediction_score_column="prediction_score",
        prediction_type="numeric",
        prediction_metrics=[
            "psi",
            "kstest",
        ],
    ),
    output_path="config.yaml",
)
```

---

## Пример настройки chi2

`chi2` используется только для categorical-фичей и возвращает p_value.

Корректные thresholds:

```python
ConfigBuildOptions(
    categorical_metrics=[
        "missing_rate",
        "psi",
        "chi2",
    ],
    global_thresholds={
        "chi2": {
            "warning": 0.05,
            "critical": 0.01,
        },
    },
)
```

В YAML:

```yaml
thresholds:
  chi2:
    warning: 0.05
    critical: 0.01
```

Интерпретация:

```text
p_value <= 0.05 -> warning
p_value <= 0.01 -> critical
```

Некорректно:

```yaml
thresholds:
  chi2:
    warning: 0.01
    critical: 0.05
```

Потому что для `chi2` меньшее значение хуже.

---

## Что делать после генерации

Рекомендуется после генерации дополнительно прогнать конфиг через pydantic-схему или стандартный загрузчик конфига.

```python
from config.parse_config import read_config

validated_config = read_config("config.yaml")
```

Это позволит поймать ошибки схемы до запуска мониторинга.

---

## Частые сценарии

### Сценарий 1. Хочу всё автоматически

```python
config = build_drift_config(df, output_path="config.yaml")
```

---

### Сценарий 2. Хочу исключить технические колонки

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        exclude_columns=[
            "id",
            "created_at",
            "updated_at",
            "request_id",
        ],
    ),
    output_path="config.yaml",
)
```

---

### Сценарий 3. Хочу свои thresholds для PSI

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        global_thresholds={
            "psi": {
                "warning": 0.1,
                "critical": 0.25,
            },
        },
    ),
    output_path="config.yaml",
)
```

Остальные thresholds будут подобраны автоматически, если автоподбор включён.

---

### Сценарий 4. Хочу другой PSI threshold для одной фичи

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        global_thresholds={
            "psi": {
                "warning": 0.1,
                "critical": 0.25,
            },
        },
        feature_thresholds={
            "age": {
                "psi": {
                    "warning": 0.05,
                    "critical": 0.12,
                },
            },
        },
    ),
    output_path="config.yaml",
)
```

---

### Сценарий 5. Хочу отключить kstest везде

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        disabled_metrics=[
            "kstest",
        ],
    ),
    output_path="config.yaml",
)
```

---

### Сценарий 6. Хочу отключить chi2 для одной categorical-фичи

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        feature_disabled_metrics={
            "country": [
                "chi2",
            ],
        },
    ),
    output_path="config.yaml",
)
```

---

### Сценарий 7. Хочу задать chi2 вручную

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        global_thresholds={
            "chi2": {
                "warning": 0.05,
                "critical": 0.01,
            },
        },
    ),
    output_path="config.yaml",
)
```

---

### Сценарий 8. Хочу менее чувствительный missing_rate

```python
config = build_drift_config(
    df,
    options=ConfigBuildOptions(
        global_thresholds={
            "missing_rate": {
                "warning": 0.01,
                "critical": 0.05,
            },
        },
    ),
    output_path="config.yaml",
)
```

---

## Рекомендации

1. Начинайте с автоматической генерации:

```python
config = build_drift_config(df, output_path="config.yaml")
```

2. Проверьте YAML глазами, особенно:

- список фичей;
- типы фичей;
- expensive metrics;
- thresholds для business-critical фичей;
- thresholds для `chi2`.

3. Для PSI обычно оставляйте стандартные thresholds:

```yaml
psi:
  warning: 0.1
  critical: 0.25
```

4. Для `chi2` используйте p_value thresholds:

```yaml
chi2:
  warning: 0.05
  critical: 0.01
```

5. Для `missing_rate` дефолтные thresholds обычно достаточно мягкие:

```yaml
missing_rate:
  warning: 0.005
  critical: 0.02
```

6. Если фичи имеют сильно разные распределения или масштабы, попробуйте:

```python
AutoThresholdSettings(
    per_feature=True
)
```

7. Если метрика дорогая или нестабильная, отключайте её глобально или точечно:

```python
ConfigBuildOptions(
    disabled_metrics=["cramer_v", "wasserstein_distance"]
)
```

---

## Краткая памятка по основным thresholds

| Метрика | Направление | Рекомендуемый дефолт |
|---|---|---|
| `missing_rate` | больше хуже | `warning=0.005`, `critical=0.02` |
| `unseen_category_rate` | больше хуже | `warning=0.005`, `critical=0.02` |
| `category_churn` | больше хуже | `warning=0.005`, `critical=0.02` |
| `psi` | больше хуже | `warning=0.1`, `critical=0.25` |
| `js_divergence` | больше хуже | `warning=0.005`, `critical=0.02` |
| `kstest` | больше хуже | `warning=0.02`, `critical=0.05` |
| `chi2` | меньше хуже | `warning=0.05`, `critical=0.01` |
| `cramer_v` | больше хуже | `warning=0.02`, `critical=0.05` |
| `cardinality_ratio` | больше хуже | `warning=0.02`, `critical=0.05` |

---