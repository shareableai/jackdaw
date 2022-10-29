import pytest

from jackdaw_ml.artefact_decorator import *
from jackdaw_ml.serializers.pickle import PickleSerializer
from tests.conftest import TEST_API_KEY

ENDPOINT = ArtefactEndpoint.remote(TEST_API_KEY)


@artefacts({PickleSerializer: "m"}, endpoint=ENDPOINT)
class ModelExample:
    def __init__(self) -> None:
        self.m = 3


@artefacts({PickleSerializer: ["m", "n"]}, endpoint=ENDPOINT)
class MultipleItem:
    def __init__(self) -> None:
        self.m = 3
        self.n = 4


@artefacts({PickleSerializer: "m"}, endpoint=ENDPOINT)
@artefacts({PickleSerializer: "m"}, endpoint=ENDPOINT)
@artefacts({PickleSerializer: "m"}, endpoint=ENDPOINT)
@artefacts({PickleSerializer: "m"}, endpoint=ENDPOINT)
@artefacts({PickleSerializer: "m"}, endpoint=ENDPOINT)
class IdempotentModelExample:
    def __init__(self) -> None:
        self.m = 3


models = [ModelExample, MultipleItem, IdempotentModelExample]


@pytest.mark.parametrize("model", models)
def test_dumps(model):
    _ = model().dumps()


@pytest.mark.parametrize("test_model", models)
def test_dump_loads(test_model):
    model = test_model()
    model.m = 400
    artefact_ids = model.dumps()
    model2 = test_model()
    model2.loads(artefact_ids)
    assert model2.m == model.m
