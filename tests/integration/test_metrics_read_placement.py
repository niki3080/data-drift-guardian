"""Статическая проверка: метрики не читаются после остановки consumer'а."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TEST_FILES = ("test_consumer_flow.py", "test_consumer_resilience.py")


def metric_read_lines(tree: ast.AST) -> set[int]:
    """Номера строк с вызовами metric_value/scrape_metrics."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name in {"metric_value", "scrape_metrics", "metric_value_strict"}:
            lines.add(node.lineno)
    return lines


def guarded_line_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Диапазоны строк внутри работающего consumer'а.

    Распознаются две формы:
      with consumer_runner(runtime): ...      (прямой вызов в with)
      runner = consumer_runner(runtime)       (фабрика в переменную)
      with runner: ...                        (затем with по имени)
    """
    # Имена переменных, которым присвоен результат consumer_runner(...).
    runner_names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "consumer_runner"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    runner_names.add(target.id)

    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                expr = item.context_expr
                is_direct = (
                    isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Name)
                    and expr.func.id == "consumer_runner"
                )
                is_named = (
                    isinstance(expr, ast.Name) and expr.id in runner_names
                )
                if is_direct or is_named:
                    ranges.append((node.lineno, node.end_lineno or node.lineno))
        if isinstance(node, ast.Try):
            finalizer_src = ast.dump(ast.Module(body=node.finalbody, type_ignores=[]))
            if "stop" in finalizer_src:
                body_start = node.body[0].lineno
                body_end = node.body[-1].end_lineno or node.body[-1].lineno
                ranges.append((body_start, body_end))
    return ranges



@pytest.mark.parametrize("filename", TEST_FILES)
def test_metrics_read_inside_running_consumer(filename: str) -> None:
    """Чтение метрик вне работающего consumer'а даёт ложный 0.0."""
    path = Path(__file__).parent / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    reads = metric_read_lines(tree)
    ranges = guarded_line_ranges(tree)

    unguarded = sorted(
        line
        for line in reads
        if not any(start <= line <= end for start, end in ranges)
    )

    assert not unguarded, (
        f"{filename}: чтение метрик вне работающего consumer'а "
        f"на строках {unguarded}. После stop() HTTP-сервер экспортёра "
        "закрыт, metric_value вернёт default=0.0."
    )