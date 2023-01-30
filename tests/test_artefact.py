import pytest

from jackdaw_ml import loads, saves
from jackdaw_ml.artefact_container import ArtefactNotFound, SupportsArtefacts
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer


@pytest.mark.local
@artefacts({PickleSerializer: "m"})
class ModelExample:
    def __init__(self) -> None:
        self.m = 3


@artefacts({PickleSerializer: ["m", "n"]})
class MultipleItem:
    def __init__(self) -> None:
        self.m = 3
        self.n = 4


@artefacts({PickleSerializer: "m"})
@artefacts({PickleSerializer: "m"})
@artefacts({PickleSerializer: "m"})
@artefacts({PickleSerializer: "m"})
@artefacts({PickleSerializer: "m"})
class IdempotentModelExample:
    def __init__(self) -> None:
        self.m = 3


models = [ModelExample, MultipleItem, IdempotentModelExample]


@pytest.mark.parametrize("model", models)
def test_attrs(model):
    assert isinstance(model(), SupportsArtefacts)


@pytest.mark.parametrize("model", models)
def test_artefact_attr(model):
    assert "m" in model().__artefact_slots__
    assert model().__artefact_slots__["m"] is PickleSerializer


@pytest.mark.parametrize("model", models)
def test_dumps(model):
    _ = saves(model())


@pytest.mark.parametrize("test_model", models)
def test_dump_loads(test_model):
    model = test_model()
    model.m = 400
    artefact_ids = saves(model)
    model2 = test_model()
    loads(model2, artefact_ids)
    assert model2.m == model.m
