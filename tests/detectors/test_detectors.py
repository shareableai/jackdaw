from typing import Any

import pytest

from jackdaw_ml.detectors import is_type


@pytest.mark.parametrize(
    "obj,obj_type",
    [
        (1, int),
        ("2", str),
        ({"1": 1, "2": 2}, dict[str, int]),
        ({"1": "1", "2": "2"}, dict[str, str]),
        ({"1": "1", "2": "2"}, dict),
        ({"1": "1", "2": "2"}, dict[Any, Any]),
        ({"1": {"a": 1}, "2": {"b": 2}}, dict[str, dict[str, int]]),
    ],
)
def test_type_check(obj, obj_type):
    assert is_type(obj, obj_type)


@pytest.mark.parametrize(
    "obj,obj_type",
    [
        (1, str),
        ("2", int),
        ({"1": 1, "2": 2}, dict[str, float]),
        ({"1": "1", "2": "2"}, dict[str, float]),
        ({"1": {"a": 1}, "2": {"b": 2}}, dict[str, dict[str, float]]),
    ],
)
def test_type_check_ne(obj, obj_type):
    assert not is_type(obj, obj_type)
