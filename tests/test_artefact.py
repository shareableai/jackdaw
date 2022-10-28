import pytest

from jackdaw_ml.artefact_decorator import *
from jackdaw_ml.serializers.pickle import PickleSerializer


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
    assert hasattr(model(), "dumps")
    assert hasattr(model(), "loads")
    assert hasattr(model(), "__artefact_slots__")


@pytest.mark.parametrize("model", models)
def test_artefact_attr(model):
    assert "m" in model().__artefact_slots__
    assert model().__artefact_slots__["m"] is PickleSerializer


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


def test_failing_missing_param():
    @artefacts({PickleSerializer: "m"})
    class ModelExample:
        def __init__(self) -> None:
            self.b = 3

    with pytest.raises(ArtefactNotFound):
        ModelExample().dumps()
