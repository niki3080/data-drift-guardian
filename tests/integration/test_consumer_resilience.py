"""Устойчивость consumer'а к невалидным сообщениям в реальном Kafka-топике.

В _process_message три ветки обработки:
  1. UnicodeDecodeError/JSONDecodeError/TypeError -> skip non-json message;
  2. ValueError из KafkaEvent.from_dict        -> skip invalid event;
  3. InvalidEventTime                          -> событие идёт в окно,
     но event_time=None и растёт invalid_event_time_rate.
Первые две ветки не увеличивают счётчик обработанных событий, третья —
увеличивает, поэтому они проверяются разными наборами метрик.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Callable

import pandas as pd
import pytest

pytestmark = pytest.mark.timeout(300)

WINDOW_SIZE = 5


def consumer_target(name: str) -> str:
    return f"drift_guardian.ingestion.consumer.{name}"


@pytest.fixture
def captured_analyses(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    """Перехватывает анализ. Возвращает (report, None): AV не запускался."""
    captured: list[pd.DataFrame] = []

    def fake_analyze(runtime: Any, current_df: pd.DataFrame):
        captured.append(current_df.copy())
        report = {
            "timestamp": "2026-09-26T06:02:00Z",
            "overall_status": "ok",
            "active_alerts": 0,
            "window_size": int(len(current_df)),
            "features": {},
        }
        return report, None

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), fake_analyze)
    return captured


# --------------------------------------------------------------------------
# Ветка 1: сообщение не декодируется или не является JSON-объектом
# --------------------------------------------------------------------------

NON_JSON_PAYLOADS = {
    "broken-json": b"{not json at all",
    "empty-body": b"",
    "invalid-utf8": b"\xff\xfe\x00\x01",
    "json-array": b"[1, 2, 3]",
    "json-scalar-string": b'"just a string"',
    "json-number": b"42",
    "json-null": b"null",
    "json-bool": b"true",
}


@pytest.mark.parametrize(
    "bad_payload",
    list(NON_JSON_PAYLOADS.values()),
    ids=list(NON_JSON_PAYLOADS),
)
def test_non_json_message_is_skipped(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        make_event,
        captured_analyses,
        consumer_runner,
        wait_until,
        bad_payload: bytes,
) -> None:
    """Не-JSON сообщение пропускается: окно закрывается только валидными.

    KafkaEvent.from_dict вообще не вызывается, событие в окно не идёт,
    поэтому для полного окна нужно ровно WINDOW_SIZE валидных событий.
    """
    payloads: list[Any] = [make_event(0), make_event(1), bad_payload]
    payloads += [make_event(i) for i in range(2, WINDOW_SIZE)]
    produce(topic, payloads)

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="consumer не восстановился после не-JSON сообщения",
        )

    assert runner.error is None
    assert len(captured_analyses[0]) == WINDOW_SIZE


def test_non_json_messages_do_not_increment_processed_counter(
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
    """Пропущенные сообщения не попадают в drift_events_processed."""
    produce(topic, list(NON_JSON_PAYLOADS.values()))
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось",
        )
        wait_until(
            lambda: metric_value("drift_events_processed") >= WINDOW_SIZE,
            message="счётчик обработанных событий не дошёл до WINDOW_SIZE",
        )
        processed = metric_value("drift_events_processed")

    # Ровно WINDOW_SIZE: битые сообщения счётчик не тронули.
    assert processed == WINDOW_SIZE


# --------------------------------------------------------------------------
# Ветка 2: JSON-объект, но KafkaEvent.from_dict бросает ValueError
# --------------------------------------------------------------------------

INVALID_EVENT_PAYLOADS = {
    "missing-event-id": {
        "event_time": "2026-09-26T06:02:00+00:00",
        "f1": 1.0,
    },
    "null-event-id": {
        "event_id": None,
        "event_time": "2026-09-26T06:02:00+00:00",
        "f1": 1.0,
    },
    "no-features": {
        "event_id": "evt-x",
        "event_time": "2026-09-26T06:02:00+00:00",
    },
    "nested-feature-dict": {
        "event_id": "evt-x",
        "event_time": "2026-09-26T06:02:00+00:00",
        "f1": {"nested": 1.0},
    },
    "list-feature": {
        "event_id": "evt-x",
        "event_time": "2026-09-26T06:02:00+00:00",
        "f1": [1.0, 2.0],
    },
}


@pytest.mark.parametrize(
    "bad_event",
    list(INVALID_EVENT_PAYLOADS.values()),
    ids=list(INVALID_EVENT_PAYLOADS),
)
def test_invalid_event_structure_is_skipped(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        make_event,
        captured_analyses,
        consumer_runner,
        wait_until,
        bad_event: dict[str, Any],
) -> None:
    """ValueError из from_dict логируется, событие пропускается."""
    payloads: list[Any] = [make_event(0), bad_event]
    payloads += [make_event(i) for i in range(1, WINDOW_SIZE)]
    produce(topic, payloads)

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="consumer не восстановился после невалидного события",
        )

    assert runner.error is None
    assert len(captured_analyses[0]) == WINDOW_SIZE


def test_only_invalid_events_never_fill_window(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        captured_analyses,
        consumer_runner,
        wait_until,
        committed_offset,
        group_id,
        metric_value: Callable[..., float],
) -> None:
    """Поток из одних невалидных сообщений: окно пустое, коммита нет."""
    produce(topic, list(NON_JSON_PAYLOADS.values()))
    produce(topic, list(INVALID_EVENT_PAYLOADS.values()))

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(
            lambda: metric_value("drift_window_size") == WINDOW_SIZE,
            message="HTTP-сервер метрик не поднялся",
        )
        assert metric_value("drift_current_window_events") == 0
        assert metric_value("drift_analysis_runs") == 0
        assert metric_value("drift_events_processed") == 0
        # Окно пустое -> _drain_partial_window ничего не делает.
        assert captured_analyses == []
    finally:
        runner.stop(reraise=False)

    assert runner.error is None
    assert captured_analyses == []
    assert committed_offset(group_id, topic) is None


def test_consumer_survives_burst_of_invalid_messages(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        make_event,
        captured_analyses,
        consumer_runner,
        wait_until,
) -> None:
    """50 битых сообщений подряд, затем валидное окно — анализ проходит."""
    payloads: list[Any] = [b"{broken" for _ in range(50)]
    payloads += [make_event(i) for i in range(WINDOW_SIZE)]
    produce(topic, payloads)

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="consumer не дошёл до валидных событий после бурста",
        )

    assert runner.error is None
    assert len(captured_analyses[0]) == WINDOW_SIZE


# --------------------------------------------------------------------------
# Ветка 3: InvalidEventTime — событие пропускается, но учитывается в метрике
# --------------------------------------------------------------------------

BAD_EVENT_TIMES = {
    "missing": None,
    "not-a-date": "definitely-not-a-timestamp",
    "empty-string": "",
    "bad-format": "26/09/2026 07:52",
}


def payload_with_event_time(index: int, event_time: Any) -> dict[str, Any]:
    """Структурно валидное событие с подменённым event_time."""
    payload: dict[str, Any] = {
        "event_id": f"evt-{index}",
        "f1": float(index),
        "f2": float(index % 3),
        "f3": float(index) * 0.5,
    }
    if event_time is not None:
        payload["event_time"] = event_time
    return payload


@pytest.mark.parametrize(
    "bad_time",
    list(BAD_EVENT_TIMES.values()),
    ids=list(BAD_EVENT_TIMES),
)
def test_invalid_event_time_is_skipped_not_windowed(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
    bad_time: Any,
) -> None:
    """InvalidEventTime: событие пропускается через continue до window.append.

    Отличие от ValueError-ветки только в record_invalid_event_time():
    событие в окно не попадает, processed не растёт, но offset
    сохраняется через store_offsets.
    """
    produce(topic, [payload_with_event_time(i, bad_time) for i in range(WINDOW_SIZE)])

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(
            lambda: metric_value("drift_invalid_event_time_rate") > 0.0,
            message="drift_invalid_event_time_rate не вырос",
        )
        # Окно пустое: все события отброшены.
        assert metric_value("drift_current_window_events") == 0
        assert metric_value("drift_events_processed") == 0
        assert metric_value("drift_analysis_runs") == 0
    finally:
        runner.stop(reraise=False)

    assert runner.error is None
    assert captured_analyses == []


def test_invalid_event_time_does_not_block_valid_events(
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
    """Битые по времени события пропускаются, окно закрывают только валидные.

    ВАЖНО: метрики читаются внутри with — после stop() HTTP-сервер
    экспортёра закрыт в finally блоке run(), scrape возвращает пустой
    словарь, и metric_value отдаёт default=0.0.
    """
    payloads: list[Any] = [
        payload_with_event_time(0, "garbage"),
        make_event(0),
        payload_with_event_time(1, "garbage"),
    ]
    payloads += [make_event(i) for i in range(1, WINDOW_SIZE)]
    produce(topic, payloads)

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось валидными событиями",
        )
        wait_until(
            lambda: metric_value("drift_events_processed") >= WINDOW_SIZE,
            message="не все валидные события приняты",
        )
        # Снимок делаем до остановки: после stop() метрики недоступны.
        processed = metric_value("drift_events_processed")

    assert len(captured_analyses[0]) == WINDOW_SIZE
    # Битые по времени события в окно не попали и не учтены как обработанные.
    assert processed == WINDOW_SIZE



def test_invalid_event_time_rate_on_live_window(
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
    """invalid_event_time_rate измеряется на живом окне после анализа.

    Метрика window-local: reset_window() после анализа её обнуляет,
    поэтому сначала закрываем окно валидными событиями (ready=True),
    затем отправляем битые по времени в остаток.
    """
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="первое окно не обработано",
        )
        wait_until(
            lambda: metric_value("drift_invalid_event_time_rate") == 0.0,
            message="rate не обнулился после reset_window",
        )

        # Остаток: 2 битых по времени + 1 валидное (окно не закроется).
        produce(topic, [
            payload_with_event_time(100, "garbage"),
            payload_with_event_time(101, "garbage"),
            make_event(WINDOW_SIZE),
        ])

        wait_until(
            lambda: metric_value("drift_invalid_event_time_rate") > 0.0,
            message="rate не вырос на остаточном окне",
        )
        rate = metric_value("drift_invalid_event_time_rate")
        assert rate == pytest.approx(0.67, abs=0.01)

    # Анализ по-прежнему один: остаток окно не закрыл.
    assert len(captured_analyses) == 1


def test_window_time_span_ignores_invalid_time_events(
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
    """span считается по валидным меткам: битые события в окно не попали."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="первое окно не обработано",
        )

        base = dt.datetime.now(dt.UTC)
        produce(topic, [
            make_event(100, event_time=base),
            payload_with_event_time(101, "garbage"),
            make_event(102, event_time=base + dt.timedelta(seconds=45)),
        ])

        wait_until(
            lambda: metric_value("drift_window_time_span_seconds") > 0.0,
            message="window_time_span не заполнен",
        )
        span = metric_value("drift_window_time_span_seconds")

    assert span == pytest.approx(45.0, abs=2.0)
    assert len(captured_analyses) == 1


# --------------------------------------------------------------------------
# Схема фич
# --------------------------------------------------------------------------


def test_extra_feature_column_does_not_crash(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        make_event,
        captured_analyses,
        consumer_runner,
        wait_until,
) -> None:
    """Лишняя фича, которой нет в reference: consumer не падает."""
    payloads: list[Any] = []
    for index in range(WINDOW_SIZE):
        event = make_event(index)
        event["f_unexpected"] = float(index) * 2
        payloads.append(event)
    produce(topic, payloads)

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось при лишней фиче",
        )

    assert runner.error is None
    assert "f_unexpected" in captured_analyses[0].columns


def test_missing_feature_produces_nan(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        make_event,
        captured_analyses,
        consumer_runner,
        wait_until,
) -> None:
    """Часть событий без f3: в DataFrame появляется NaN, падения нет."""
    payloads: list[Any] = []
    for index in range(WINDOW_SIZE):
        event = make_event(index)
        if index % 2 == 0:
            event.pop("f3")
        payloads.append(event)
    produce(topic, payloads)

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось при отсутствующей фиче",
        )

    assert runner.error is None
    frame = captured_analyses[0]
    assert len(frame) == WINDOW_SIZE
    assert frame["f3"].isna().sum() == 3


def test_string_and_null_feature_values_accepted(
        runtime,
        no_signal_handlers,
        produce,
        topic,
        make_event,
        captured_analyses,
        consumer_runner,
        wait_until,
) -> None:
    """Строка и null — JSON-скаляры, события принимаются."""
    payloads: list[Any] = []
    for index in range(WINDOW_SIZE):
        event = make_event(index)
        event["f2"] = "category_a"
        event["f1"] = None
        payloads.append(event)
    produce(topic, payloads)

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось",
        )

    assert runner.error is None
    frame = captured_analyses[0]
    assert frame["f2"].iloc[0] == "category_a"
    assert frame["f1"].isna().all()


# --------------------------------------------------------------------------
# Падение анализа: окно теряется, но offset продвигается
# --------------------------------------------------------------------------


def test_analysis_failure_still_commits_offset(
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
    """При падении анализа окно очищается и offset коммитится.

    В _handle_full_window блок except Exception логирует analysis failed,
    но window.clear() и consumer.commit() вызываются безусловно — окно
    теряется, at-least-once для анализа не гарантируется.
    """
    attempts: list[int] = []

    def failing_analyze(_runtime: Any, current_df: pd.DataFrame):
        attempts.append(len(current_df))
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), failing_analyze)

    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(lambda: len(attempts) >= 1, message="анализ не был вызван")
        wait_until(
            lambda: committed_offset(group_id, topic) == WINDOW_SIZE,
            message="offset не закоммичен после падения анализа",
        )
        # Окно очищено несмотря на ошибку.
        wait_until(
            lambda: metric_value("drift_current_window_events") == 0,
            message="окно не очистилось после падения анализа",
        )
        assert metric_value("drift_analysis_runs") == 0
    finally:
        runner.stop(reraise=False)

    assert runner.error is None
    assert len(attempts) == 1

def test_window_is_lost_not_retried_after_failure(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    make_event,
    consumer_runner,
    wait_until,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Упавшее окно не перечитывается: второе окно содержит новые события.

    Ключевое следствие безусловного window.clear() + commit() в
    _handle_full_window: события 0..4 теряются навсегда, второй анализ
    получает события 5..9, а не повтор первого окна.
    """
    calls: list[list[float]] = []
    succeeded: list[pd.DataFrame] = []

    def flaky_analyze(_runtime: Any, current_df: pd.DataFrame):
        calls.append(current_df["f1"].tolist())
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        succeeded.append(current_df.copy())
        return (
            {
                "timestamp": "2026-09-26T06:04:00Z",
                "overall_status": "ok",
                "active_alerts": 0,
                "window_size": int(len(current_df)),
                "features": {},
            },
            None,
        )

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), flaky_analyze)

    produce(topic, [make_event(i) for i in range(WINDOW_SIZE * 2)])

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(
            lambda: len(succeeded) >= 1,
            message="consumer не обработал второе окно после падения первого",
        )
    finally:
        runner.stop(reraise=False)

    assert runner.error is None
    assert len(calls) == 2

    first_window, second_window = calls[0], calls[1]
    assert first_window == [0.0, 1.0, 2.0, 3.0, 4.0]
    # Не повтор первого окна — упавшие события потеряны.
    assert second_window == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert set(first_window).isdisjoint(second_window)


def test_export_failure_does_not_kill_consumer(
    runtime,
    no_signal_handlers,
    produce,
    topic,
    group_id,
    make_event,
    committed_offset,
    consumer_runner,
    wait_until,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Падение экспорта отчёта не останавливает цикл потребления.

    _export_completed_analysis вызывается внутри того же try/except,
    что и анализ, поэтому исключение из него тоже приводит к clear+commit.
    """
    export_calls: list[int] = []

    def failing_export(*_args: Any, **_kwargs: Any) -> None:
        export_calls.append(1)
        raise RuntimeError("exporter is down")

    monkeypatch.setattr(consumer_target("_export_completed_analysis"), failing_export)

    def ok_analyze(_runtime: Any, current_df: pd.DataFrame):
        return (
            {
                "timestamp": "2026-09-26T06:04:00Z",
                "overall_status": "ok",
                "active_alerts": 0,
                "window_size": int(len(current_df)),
                "features": {},
            },
            None,
        )

    monkeypatch.setattr(consumer_target("analyze_current_dataframe"), ok_analyze)

    produce(topic, [make_event(i) for i in range(WINDOW_SIZE * 2)])

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(
            lambda: len(export_calls) >= 2,
            message="consumer остановился после первой ошибки экспорта",
        )
        wait_until(
            lambda: committed_offset(group_id, topic) == WINDOW_SIZE * 2,
            message="offset не продвинулся при падающем экспорте",
        )
    finally:
        runner.stop(reraise=False)

    assert runner.error is None


# --------------------------------------------------------------------------
# Остановка: _drain_partial_window
# --------------------------------------------------------------------------


def test_stop_with_partial_window_does_not_analyze(
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
    """Остановка на недозаполненном окне: анализа и коммита нет.

    В run() нет слива частичного окна: события останутся незакоммиченными
    (store_offsets хранит в памяти, commit — только после полного окна) и
    будут перечитаны после рестарта группы. At-least-once без потерь.
    """
    partial = WINDOW_SIZE - 2
    produce(topic, [make_event(i) for i in range(partial)])

    runner = consumer_runner(runtime)
    with runner:
        wait_until(
            lambda: metric_value("drift_current_window_events") == partial,
            message="события не попали в окно",
        )
        assert captured_analyses == []

    assert runner.error is None
    # Анализа не было: окно не закрылось, слива нет.
    assert captured_analyses == []
    # Коммита нет: события перечитаются после рестарта.
    assert committed_offset(group_id, topic) is None



def test_stop_with_empty_window_does_not_analyze(
    runtime,
    no_signal_handlers,
    topic,
    group_id,
    captured_analyses,
    committed_offset,
    consumer_runner,
    wait_until,
    metric_value: Callable[..., float],
) -> None:
    """Остановка на пустом окне: _drain_partial_window не запускает анализ."""
    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(
            lambda: metric_value("drift_window_size") == WINDOW_SIZE,
            message="HTTP-сервер метрик не поднялся",
        )
        assert metric_value("drift_current_window_events") == 0
    finally:
        runner.stop()

    assert captured_analyses == []
    assert committed_offset(group_id, topic) is None


def test_stop_after_full_window_does_not_double_analyze(
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
    """Ровно WINDOW_SIZE событий: один анализ в цикле, слив не дублирует."""
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    runner = consumer_runner(runtime)
    runner.start()
    try:
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не обработано",
        )
        wait_until(
            lambda: committed_offset(group_id, topic) == WINDOW_SIZE,
            message="offset не закоммичен",
        )
    finally:
        runner.stop()

    # window.clear() отработал, слив нечего анализировать.
    assert len(captured_analyses) == 1



# --------------------------------------------------------------------------
# Смешанная нагрузка
# --------------------------------------------------------------------------


def test_mixed_valid_and_broken_stream_stays_consistent(
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
    """Чередование валидных, не-JSON и структурно битых сообщений.

    10 валидных событий, между ними мусор: должно закрыться два окна,
    счётчик обработанных равен только числу валидных.
    """
    payloads: list[Any] = []
    for index in range(WINDOW_SIZE * 2):
        payloads.append(make_event(index))
        if index % 3 == 0:
            payloads.append(b"{broken json")
        if index % 4 == 0:
            payloads.append({"event_time": "2026-09-26T06:04:00+00:00", "f1": 1.0})

    produce(topic, payloads)

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 2,
            message="не закрылись два окна в смешанном потоке",
        )
        wait_until(
            lambda: metric_value("drift_events_processed") == WINDOW_SIZE * 2,
            message="счётчик обработанных не совпал с числом валидных событий",
        )

    assert [len(df) for df in captured_analyses[:2]] == [WINDOW_SIZE, WINDOW_SIZE]
    # Окна не пересекаются: f1 монотонен по индексу.
    assert captured_analyses[0]["f1"].max() < captured_analyses[1]["f1"].min()


def test_invalid_time_events_mixed_with_broken_messages(
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
    """Все три ветки в одном потоке: окно закрывают только валидные события.

    from_dict валидирует event_time до фич: payload без event_time уходит
    через InvalidEventTime даже с битыми фичами. Для ValueError-ветки
    нужен payload с валидным event_time и нескалярной фичей.
    """
    valid_iso = dt.datetime.now(dt.UTC).isoformat()

    # Окно 1: 5 валидных событий, между ними мусор всех трёх веток.
    payloads: list[Any] = [
        b"\xff\xfe",                                              # не-JSON
        make_event(0),
        {"event_id": "x", "f1": [1, 2]},                          # InvalidEventTime (нет event_time)
        make_event(1),
        b"[]",                                                    # не JSON-объект
        make_event(2),
        {"event_id": "y", "event_time": valid_iso, "f1": [1, 2]}, # ValueError (битая фича)
        payload_with_event_time(10, "garbage"),                   # InvalidEventTime (не парсится)
        b"{broken",
        make_event(3),
        make_event(4),
    ]
    produce(topic, payloads)

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="окно не закрылось валидными событиями",
        )
        wait_until(
            lambda: metric_value("drift_events_processed") >= WINDOW_SIZE,
            message="не все валидные события приняты",
        )
        processed = metric_value("drift_events_processed")

    # Ровно 5 валидных: мусор всех веток счётчик не тронул.
    assert processed == WINDOW_SIZE
    assert len(captured_analyses[0]) == WINDOW_SIZE


def test_all_branches_tracked_on_residual_window(
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
    """invalid_event_time_rate растёт от смешанного потока на живом окне.

    Метрика window-local: после анализа reset_window() её обнуляет,
    поэтому измеряем на остаточном окне после закрытия первого.
    """
    produce(topic, [make_event(i) for i in range(WINDOW_SIZE)])

    with consumer_runner(runtime):
        wait_until(
            lambda: len(captured_analyses) >= 1,
            message="первое окно не обработано",
        )

        # Остаток: битое по времени + валидное (окно не закроется).
        produce(topic, [
            payload_with_event_time(20, "garbage"),
            make_event(WINDOW_SIZE),
        ])

        wait_until(
            lambda: metric_value("drift_invalid_event_time_rate") > 0.0,
            message="invalid_event_time_rate не вырос на остатке",
        )
        rate = metric_value("drift_invalid_event_time_rate")
        print(f"invalid_event_time_rate на остатке: {rate}")

    assert len(captured_analyses) == 1



