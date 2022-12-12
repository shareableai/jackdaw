import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.remote
@pytest.mark.parametrize('model_under_test', [
    lazy_fixture('simple_model_example'),
    lazy_fixture('multiple_item_example'),
    lazy_fixture('idempotent_model_example')
])
def test_dumps(model_under_test):
    _ = model_under_test().dumps()


@pytest.mark.remote
@pytest.mark.parametrize('model_under_test', [
    lazy_fixture('simple_model_example'),
    lazy_fixture('multiple_item_example'),
    lazy_fixture('idempotent_model_example')
])
def test_dump_loads(model_under_test):
    model = model_under_test()
    model.m = 400
    artefact_ids = model.dumps()
    model2 = model_under_test()
    model2.loads(artefact_ids)
    assert model2.m == model.m
