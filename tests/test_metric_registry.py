import pandas as pd

from src.drift_guardian.analyzer.regestry import metric_registry


def dummy_reference_dict():
    return {
        "cat_ref": {},
        "num_ref": {},
        "sample": pd.DataFrame(),
        "preds_ref": {},
    }


def test_metric_registry_is_dict():
    assert isinstance(metric_registry.METRIC_REGISTRY, dict)


def test_register_adds_function_to_registry():
    original_registry = metric_registry.METRIC_REGISTRY.copy()
    metric_registry.METRIC_REGISTRY.clear()

    try:
        metric_name = "__pytest_test_metric__"

        def metric_fn(reference_dict, current, **kwargs):
            return 1.0

        result = metric_registry.register(metric_name)(metric_fn)

        assert metric_name in metric_registry.METRIC_REGISTRY
        assert metric_registry.METRIC_REGISTRY[metric_name] is metric_fn
        assert result is metric_fn

    finally:
        metric_registry.METRIC_REGISTRY.clear()
        metric_registry.METRIC_REGISTRY.update(original_registry)


def test_register_works_as_decorator():
    original_registry = metric_registry.METRIC_REGISTRY.copy()
    metric_registry.METRIC_REGISTRY.clear()

    try:
        metric_name = "__pytest_decorator_metric__"

        @metric_registry.register(metric_name)
        def metric_fn(reference_dict, current, **kwargs):
            return 2.5

        assert metric_name in metric_registry.METRIC_REGISTRY
        assert metric_registry.METRIC_REGISTRY[metric_name] is metric_fn

        result = metric_registry.METRIC_REGISTRY[metric_name](
            dummy_reference_dict(),
            pd.Series([1, 2, 3]),
        )

        assert result == 2.5

    finally:
        metric_registry.METRIC_REGISTRY.clear()
        metric_registry.METRIC_REGISTRY.update(original_registry)


def test_register_returns_original_function():
    original_registry = metric_registry.METRIC_REGISTRY.copy()
    metric_registry.METRIC_REGISTRY.clear()

    try:
        metric_name = "__pytest_return_metric__"

        def metric_fn(reference_dict, current, **kwargs):
            return 10.0

        decorated_fn = metric_registry.register(metric_name)(metric_fn)

        assert decorated_fn is metric_fn

    finally:
        metric_registry.METRIC_REGISTRY.clear()
        metric_registry.METRIC_REGISTRY.update(original_registry)


def test_register_overwrites_existing_metric():
    original_registry = metric_registry.METRIC_REGISTRY.copy()
    metric_registry.METRIC_REGISTRY.clear()

    try:
        metric_name = "__pytest_overwrite_metric__"

        def first_metric_fn(reference_dict, current, **kwargs):
            return 1.0

        def second_metric_fn(reference_dict, current, **kwargs):
            return 2.0

        metric_registry.register(metric_name)(first_metric_fn)
        metric_registry.register(metric_name)(second_metric_fn)

        assert metric_registry.METRIC_REGISTRY[metric_name] is second_metric_fn

        result = metric_registry.METRIC_REGISTRY[metric_name](
            dummy_reference_dict(),
            pd.Series([1, 2, 3]),
        )

        assert result == 2.0

    finally:
        metric_registry.METRIC_REGISTRY.clear()
        metric_registry.METRIC_REGISTRY.update(original_registry)


def test_register_keeps_multiple_metrics_independently():
    original_registry = metric_registry.METRIC_REGISTRY.copy()
    metric_registry.METRIC_REGISTRY.clear()

    try:
        first_metric = "__pytest_first_metric__"
        second_metric = "__pytest_second_metric__"

        def first_metric_fn(reference_dict, current, **kwargs):
            return 1.0

        def second_metric_fn(reference_dict, current, **kwargs):
            return 2.0

        metric_registry.register(first_metric)(first_metric_fn)
        metric_registry.register(second_metric)(second_metric_fn)

        assert metric_registry.METRIC_REGISTRY[first_metric] is first_metric_fn
        assert metric_registry.METRIC_REGISTRY[second_metric] is second_metric_fn
        assert len(metric_registry.METRIC_REGISTRY) == 2

    finally:
        metric_registry.METRIC_REGISTRY.clear()
        metric_registry.METRIC_REGISTRY.update(original_registry)


def test_registered_metric_receives_reference_current_and_kwargs():
    original_registry = metric_registry.METRIC_REGISTRY.copy()
    metric_registry.METRIC_REGISTRY.clear()

    try:
        metric_name = "__pytest_kwargs_metric__"
        reference = dummy_reference_dict()
        current = pd.Series([1, 2, 3])

        def metric_fn(reference_dict, current_arg, **kwargs):
            assert reference_dict is reference
            assert current_arg.equals(current)
            assert kwargs["threshold"] == 0.5
            return 123.0

        metric_registry.register(metric_name)(metric_fn)

        result = metric_registry.METRIC_REGISTRY[metric_name](
            reference,
            current,
            threshold=0.5,
        )

        assert result == 123.0

    finally:
        metric_registry.METRIC_REGISTRY.clear()
        metric_registry.METRIC_REGISTRY.update(original_registry)
