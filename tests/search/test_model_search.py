from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml import loads
from jackdaw_ml import saves
from jackdaw_ml.search import Searcher
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer

from math import sqrt


@artefacts({PickleSerializer: ["model"]})
class BasicModelExample:
    model: int

    def __init__(self):
        self.model = 4

    def __call__(self, X: int) -> int:
        return X * self.model

    def eval(self, y_hat, y) -> float:
        rmse = sqrt((y_hat - y) ** 2)

        return rmse


def test_search_models_by_name():
    x = BasicModelExample()
    x.model = 10
    model_id = saves(x)
    models = (
        Searcher(ArtefactEndpoint.default()).with_name(str(BasicModelExample)).models()
    )
    saved_model_id = next(iter(models))
    y = BasicModelExample()
    loads(y, saved_model_id)
    assert (
        saved_model_id.artefact_schema_id.as_string()
        == model_id.artefact_schema_id.as_string()
    )
    assert y.model == x.model


def test_search_models_by_metric():
    x = BasicModelExample()
    x.model = 10
    model_id = saves(x)
    models = (
        Searcher(ArtefactEndpoint.default()).with_name(str(BasicModelExample)).models()
    )
    saved_model_id = next(iter(models))
    y = BasicModelExample()
    loads(y, saved_model_id)
    assert (
        saved_model_id.artefact_schema_id.as_string()
        == model_id.artefact_schema_id.as_string()
    )
    assert y.model == x.model
