"""Интеграционные тесты основного потока consumer'а против реальной Kafka."""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

import pandas as pd
import pytest

pytestmark = pytest.mark.timeout(300)

WINDOW_SIZE = 5


def consumer_target(name: str) -> str:
    """Патчим имя в namespace consumer'а, куда оно импортировано."""
    return f"drift_guardian.ingestion.consumer.{name}"


@pytest.fixture
def captured_analyses(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    """Перехватывает analyze_current_dataframe вместо реального Core-анализа.

    Возвращает (report, None): None означает, что AV не запускался,
    что согласуется с adversarial_validation.enabled=false в конфиге.
    """
    captured: list[pd.DataFrame] = []

    def fake_analyze(runtime: Any, current_df: pd.DataFrame, **_kwargs: Any):
        captured.append(current_df.copy())
        runtime._av_executed_last_analysis = False
        report = {
            "timestamp": "2026-09-26T05:57:00Z",
            "overall_status": "ok",
            "active_alerts": 0,
            "window_size": int(len(current_df)),
            "features": {},
        }
        return report, None

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), fake_analyze)
    return captured

# --------------------------------------------------------------------------
# Базовый happy path
# --------------------------------------------------------------------------


def test_full_window_triggers_analysis(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
) -> None:
    """Полное окно запускает анализ один раз с нужным числом строк."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="анализ не выполнился после заполнения окна",
        )

    assert len(captured_analyses) == 1
    assert len(captured_analyses[0]) == WINDOW_SIZE


def test_window_dataframe_contains_flat_features(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
) -> None:
    """DataFrame окна содержит колонки фич, а не event_id/event_time."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(lambda: len(captured_analyses) >= 1, message="анализа нет")

    frame = captured_analyses[0]
    for name in ("f1", "f2", "f3"):
        assert name in frame.columns, f"колонка {name} отсутствует"
    assert "event_id" not in frame.columns
    assert "event_time" not in frame.columns


def test_offset_committed_after_successful_analysis(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
) -> None:
    """После успешного анализа offset коммитится на все обработанные события."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: committed_offset(group_id, topic) == WINDOW_SIZE,
            message="offset не закоммичен после успешного анализа",
        )


# --------------------------------------------------------------------------
# Неполное окно
# --------------------------------------------------------------------------


def test_partial_window_does_not_trigger_analysis(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """Неполное окно: события приняты, анализа и коммита нет.

    Ждём drift_events_processed вместо sleep: счётчик растёт в
    record_processed_event() на каждое принятое событие.
    """
    partial = WINDOW_SIZE - 1
    produce(topic, [make_event(i) for i in range(partial)])

    with consumer_runner(runtime):
        wait_until(
            lambda: metric_value("drift_events_processed") >= partial,
            message=f"consumer не принял все {partial} событий",
        )
        assert metric_value("drift_current_window_events") == partial
        assert metric_value("drift_analysis_runs") == 0

    assert captured_analyses == []
    assert committed_offset(group_id, topic) is None


def test_window_completed_by_later_batch(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """Окно, добитое второй партией событий, запускает анализ."""
    with consumer_runner(runtime):
        produce(topic, [make_event(i) for i in range(WINDOW_SIZE - 2)])
        wait_until(
            lambda: metric_value("drift_current_window_events") == WINDOW_SIZE - 2,
            message="первая партия не попала в окно",
        )
        assert captured_analyses == []

        produce(topic, [make_event(i) for i in range(WINDOW_SIZE - 2, WINDOW_SIZE)])
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось после второй партии",
        )
        wait_until(
            lambda: committed_offset(group_id, topic) == WINDOW_SIZE,
            message="offset не закоммичен",
        )

    assert len(captured_analyses[0]) == WINDOW_SIZE


# --------------------------------------------------------------------------
# Несколько окон подряд
# --------------------------------------------------------------------------


def test_two_full_windows_processed_sequentially(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """Два окна подряд: два анализа, offset на 2 * WINDOW_SIZE."""
    total = WINDOW_SIZE * 2
    produce(topic, [make_event(i) for i in range(total)])

    with consumer_runner(runtime):
        wait_until(
            lambda: metric_value("drift_analysis_runs") >= 2,
            message="второе окно не обработано",
        )
        wait_until(
            lambda: committed_offset(group_id, topic) == total,
            message="offset не догнал оба окна",
        )
        assert metric_value("drift_events_processed") == total

    assert [len(df) for df in captured_analyses[:2]] == [WINDOW_SIZE, WINDOW_SIZE]


def test_windows_do_not_overlap(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
) -> None:
    """Окно очищается: второй анализ не содержит событий первого."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE * 2)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 2,
            message="нужно минимум два анализа",
        )

    first, second = captured_analyses[0], captured_analyses[1]
    assert len(first) == WINDOW_SIZE == len(second)
    # f1 монотонно растёт по индексу события — окна не должны пересекаться.
    assert first["f1"].max() < second["f1"].min()


def test_window_counter_resets_after_analysis(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """После анализа окно очищается, drift_analysis_runs растёт."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: metric_value("drift_analysis_runs") >= 1,
            message="drift_analysis_runs не увеличился",
        )
        wait_until(
            lambda: metric_value("drift_current_window_events") == 0,
            message="drift_current_window_events не сбросился",
        )
        assert metric_value("drift_window_size") == WINDOW_SIZE


def test_remainder_stays_in_window_after_analysis(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """WINDOW_SIZE + 2 события: анализ один, остаток ждёт в окне."""
    total = WINDOW_SIZE + 2
    produce(topic, [make_event(i) for i in range(total)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="первое окно не обработано",
        )
        wait_until(
            lambda: metric_value("drift_events_processed") >= total,
            message="не все события приняты",
        )
        assert metric_value("drift_current_window_events") == 2
        assert metric_value("drift_analysis_runs") == 1
        # Коммит только за закрытое окно, остаток не закоммичен.
        assert committed_offset(group_id, topic) == WINDOW_SIZE

    assert len(captured_analyses) == 1


# --------------------------------------------------------------------------
# Ошибки анализа и восстановление
# --------------------------------------------------------------------------
def test_analysis_failure_still_advances_offset(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Падение анализа: окно теряется, offset коммитится, consumer живёт.

    Семантика после правки run(): анализ и экспорт обёрнуты в try/except,
    LOGGER.exception логирует падение, затем безусловно выполняются
    commit() и window.clear() — упавшее окно не перечитывается.
    """
    attempts: list[int] = []

    def failing_analyze(_runtime: Any, current_df: pd.DataFrame, **_kwargs: Any):
        attempts.append(len(current_df))
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), failing_analyze)

    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(lambda: len(attempts) >= 1, message="анализ не был вызван")
        # Успешных анализов нет — счётчик не растёт.
        assert metric_value("drift_analysis_runs") == 0
        # Окно сброшено несмотря на падение.
        wait_until(
            lambda: metric_value("drift_current_window_events") == 0,
            message="окно не очистилось после падения анализа",
        )
        # Offset продвинут: окно потеряно, consumer продолжает.
        wait_until(
            lambda: committed_offset(group_id, topic) == WINDOW_SIZE,
            message="offset не закоммичен после падения анализа",
        )
    finally:
        runner.stop(reraise=False)

    assert runner.error is None
    assert len(attempts) == 1

def test_recovery_after_transient_analysis_failure(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[float]] = []
    succeeded: list[pd.DataFrame] = []

    def flaky_analyze(_runtime: Any, current_df: pd.DataFrame):
        calls.append(current_df["f1"].tolist())
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        succeeded.append(current_df.copy())
        return (
            {
                "timestamp": "2026-09-26T07:00:00Z",
                "overall_status": "ok",
                "active_alerts": 0,
                "window_size": int(len(current_df)),
                "features": {},
            },
            None,
        )

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), flaky_analyze)

    # Проверяем, что патч действительно подменил то имя,
    # которое consumer вызывает в _handle_full_window.
    from drift_guardian.ingestion import consumer as consumer_module


    assert consumer_module.analyze_current_dataframe is flaky_analyze, (
        "патч не применился: consumer вызывает другую ссылку на функцию"
    )

    # STOP должен быть сброшен: иначе run() выйдет из цикла на первой проверке.
    assert not consumer_module.STOP.is_set(), (
        "STOP установлен до старта — состояние протекло из предыдущего теста"
    )

    produce(topic, [make_event(i) for i in range(WINDOW_SIZE * 2)])

    runner = consumer_runner(runtime)
    runner.start()

    assert not consumer_module.STOP.is_set(), "STOP установлен сразу после start()"
    try:
        wait_until(
            lambda: metric_value("drift_events_processed") > 0,
            timeout=30.0,
            message="ни одно событие не принято",
        )
        wait_until(
            lambda: len(calls) >= 1,
            timeout=30.0,
            message=(
                f"анализ не вызван: processed="
                f"{metric_value('drift_events_processed')}, "
                f"window={metric_value('drift_current_window_events')}, "
                f"runs={metric_value('drift_analysis_runs')}"
            ),
        )
        wait_until(
            lambda: len(succeeded) >= 1,
            timeout=30.0,
            message=f"второе окно не обработано, calls={len(calls)}",
        )
    finally:
        runner.stop(reraise=False)

    assert len(calls) == 2
    assert calls[0] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calls[1] == [5.0, 6.0, 7.0, 8.0, 9.0]




# --------------------------------------------------------------------------
# Экспорт метрик
# --------------------------------------------------------------------------


def test_stream_metrics_are_exported(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
    scrape_metrics: Callable[[], dict[str, float]],
) -> None:
    """Stream-метрики появляются в /metrics после анализа."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="анализ не выполнился",
        )
        wait_until(
            lambda: "drift_event_time_lag_seconds" in scrape_metrics(),
            message="stream-метрики не экспортированы",
        )
        snapshot = scrape_metrics()

    for name in (
        "drift_event_time_lag_seconds",
        "drift_window_time_span_seconds",
        "drift_invalid_event_time_rate",
        "drift_late_event_rate",
        "drift_overall_status",
        "drift_last_analysis_age_seconds",
    ):
        assert name in snapshot, f"метрика {name} отсутствует в /metrics"

    # Отчёт-заглушка имеет overall_status=ok, что маппится в 0.
    assert snapshot["drift_overall_status"] == 0


def test_av_unavailable_when_disabled_in_config(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
    metric_value_strict: Callable[[str], float],
) -> None:
    """AV выключен в конфиге: экспортёр сообщает 'недоступен' явными сериями.

    Экспортёр инициализирует AV-метрики в __init__ (drift_av_available=0,
    drift_av_status=-1), поэтому серии существуют всегда. Строгая фикстура
    гарантирует, что /metrics жив и метрика на месте: мёртвый сервер или
    пропавшая серия теперь падают с явной ошибкой, а не маскируются
    default'ом (-999/-1), как в прежней версии теста.
    """
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="анализ не выполнился",
        )
        assert metric_value_strict("drift_av_available") == 0.0
        assert metric_value_strict("drift_av_status") == -1.0


# --------------------------------------------------------------------------
# Временные метки событий
# --------------------------------------------------------------------------


def test_late_events_increase_late_rate(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """late_event_rate растёт на живом (незакрытом) окне с late-событиями.

    Метрика window-local: после анализа reset_window() обнуляет её.
    Поэтому измеряем на остаточном окне — первом валидным окном
    закрываем анализ (ready=True), затем отправляем 2 late + 1 свежее
    событие, которые окно не закрывают.
    """
    now = dt.datetime.now(dt.UTC)
    stale = now - dt.timedelta(seconds=600)  # >> LATE_EVENT_THRESHOLD_SECONDS=60

    # Окно 1: валидные события, закрывает анализ.
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="первое окно не обработано",
        )

        # Остаточное окно: 2 late + 1 свежее, is_full не наступит.
        produce(topic, [
            make_event(WINDOW_SIZE, event_time=stale),
            make_event(WINDOW_SIZE + 1, event_time=stale),
            make_event(WINDOW_SIZE + 2, event_time=now),
        ])

        wait_until(
            lambda: metric_value("drift_events_processed") >= WINDOW_SIZE + 3,
            message="остаточные события не приняты",
        )
        wait_until(
            lambda: metric_value("drift_late_event_rate") > 0.0,
            message="drift_late_event_rate остался нулевым",
        )
        rate = metric_value("drift_late_event_rate")

    assert len(captured_analyses) == 1


def test_window_time_span_reflects_event_times(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    captured_analyses,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """window_time_span равен размаху event_time живого (незакрытого) окна.

    span — window-local метрика: после анализа window.clear() и
    stream.reset_window() обнуляют её в последнем снапшоте. Поэтому
    измеряем на остаточном окне: первое окно закрываем валидными
    событиями (анализ, ready=True), затем отправляем WINDOW_SIZE - 2
    событий с шагом 30 секунд — окно не закрывается, и последний
    update_stream на append экспортирует span по живым событиям.
    """
    base = dt.datetime.now(dt.UTC)

    # Окно 1: закрывает анализ, переводит stream-метрики в ready=True.
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="первое окно не обработано",
        )

        # Окно 2 (остаточное): 3 события с шагом 30с, is_full не наступит.
        residual = WINDOW_SIZE - 2
        produce(topic, [
            make_event(WINDOW_SIZE + i, event_time=base + dt.timedelta(seconds=i * 30))
            for i in range(residual)
        ])

        wait_until(
            lambda: metric_value("drift_events_processed") >= WINDOW_SIZE + residual,
            message="остаточные события не приняты",
        )
        wait_until(
            lambda: metric_value("drift_window_time_span_seconds") > 0.0,
            message="window_time_span не заполнен",
        )
        span = metric_value("drift_window_time_span_seconds")

    # Размах остатка: base, +30s, +60s → 60 секунд.
    assert span == pytest.approx(60.0, abs=2.0)
    # Анализ по-прежнему один — остаток не закрыл окно.
    assert len(captured_analyses) == 1

