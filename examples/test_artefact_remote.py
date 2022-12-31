import pytest
from pytest_lazyfixture import lazy_fixture

from jackdaw_ml.loads import loads
from jackdaw_ml.saves import saves


@pytest.mark.remote
@pytest.mark.parametrize(
    "model_under_test",
    [
        lazy_fixture("simple_model_example"),
        lazy_fixture("multiple_item_example"),
        lazy_fixture("idempotent_model_example"),
    ],
)
def test_dumps(model_under_test):
    _ = model_under_test().dumps()


@pytest.mark.remote
@pytest.mark.parametrize(
    "model_under_test",
    [
        lazy_fixture("simple_model_example"),
        lazy_fixture("multiple_item_example"),
        lazy_fixture("idempotent_model_example"),
    ],
)
def test_dump_loads(model_under_test):
    model = model_under_test()
    model.m = 400
    model_id = saves(model)
    model2 = model_under_test()
    loads(model2, model_id)
    assert model2.m == model.m
