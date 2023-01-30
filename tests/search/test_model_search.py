from math import sqrt

import pytest
from artefact_link import PyModelSearchResult, search_by_model_id

from jackdaw_ml import loads, saves
from jackdaw_ml.artefact_decorator import artefacts, format_class_name
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.search import Searcher
from jackdaw_ml.serializers.pickle import PickleSerializer
from tests.conftest import TEST_API_KEY


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


@pytest.mark.remote
def test_search_models_by_name_remotely():
    @artefacts(
        {PickleSerializer: ["model"]}, endpoint=ArtefactEndpoint.remote(TEST_API_KEY)
    )
    class BasicRemoteModel:
        model: int

        def __init__(self):
            self.model = 20

        def __call__(self, X: int) -> int:
            return X * self.model

        def eval(self, y_hat, y) -> float:
            rmse = sqrt((y_hat - y) ** 2)

            return rmse

    x = BasicRemoteModel()
    x.model = 10
    model_id = saves(x)

    models = (
        Searcher(ArtefactEndpoint.remote(TEST_API_KEY))
        .with_name(str(BasicRemoteModel))
        .models()
    )
    for (idx, result) in enumerate(models):
        y = BasicRemoteModel()
        loads(y, result.model_id)
        assert (
            result.model_id.artefact_schema_id.as_string()
            == model_id.artefact_schema_id.as_string()
        )
        assert y.model == x.model


def test_lookup_name():
    @artefacts({PickleSerializer: ["model"]}, endpoint=ArtefactEndpoint.default())
    class LookupModelByNameTest:
        model: int

        def __init__(self):
            self.model = 20

        def __call__(self, X: int) -> int:
            return X * self.model

        def eval(self, y_hat, y) -> float:
            rmse = sqrt((y_hat - y) ** 2)

            return rmse

    y = LookupModelByNameTest()
    model_id = saves(y)
    model = next(
        iter(
            Searcher(ArtefactEndpoint.default())
            .with_name(format_class_name(str(LookupModelByNameTest)))
            .models()
        )
    )
    looked_up_model_id = search_by_model_id(
        ArtefactEndpoint.default().endpoint,
        model.vcs_info.short_sha,
        model.model_id.short_schema_id,
        model.model_id.name,
    )
    assert (
        model_id.artefact_schema_id.as_string()
        == looked_up_model_id.artefact_schema_id.as_string()
    )


@pytest.mark.remote
def test_lookup_name_remote():
    @artefacts(
        {PickleSerializer: ["model"]}, endpoint=ArtefactEndpoint.remote(TEST_API_KEY)
    )
    class LookupModelByNameTest:
        model: int

        def __init__(self):
            self.model = 20

        def __call__(self, X: int) -> int:
            return X * self.model

        def eval(self, y_hat, y) -> float:
            rmse = sqrt((y_hat - y) ** 2)

            return rmse

    y = LookupModelByNameTest()
    model_id = saves(y)
    model = next(
        iter(
            Searcher(ArtefactEndpoint.remote(TEST_API_KEY))
            .with_name(format_class_name(str(LookupModelByNameTest)))
            .models()
        )
    )
    looked_up_model_id = search_by_model_id(
        ArtefactEndpoint.remote(TEST_API_KEY).endpoint,
        model.vcs_info.short_sha,
        model.model_id.short_schema_id,
        model.model_id.name,
    )
    assert (
        model_id.artefact_schema_id.as_string()
        == looked_up_model_id.artefact_schema_id.as_string()
    )


def test_search_models_by_name():
    x = BasicModelExample()
    x.model = 10
    model_id = saves(x)
    models = (
        Searcher(ArtefactEndpoint.default())
        .with_name(format_class_name(str(BasicModelExample)))
        .models()
    )
    model_result: PyModelSearchResult = next(iter(models))
    y = BasicModelExample()
    loads(y, model_result.model_id)
    assert (
        model_result.model_id.artefact_schema_id.as_string()
        == model_id.artefact_schema_id.as_string()
    )
    assert y.model == x.model


def test_search_models_by_metric():
    x = BasicModelExample()
    x.model = 10
    model_id = saves(x)
    models = (
        Searcher(ArtefactEndpoint.default())
        .with_name(format_class_name(str(BasicModelExample)))
        .models()
    )
    model_result: PyModelSearchResult = next(iter(models))
    y = BasicModelExample()
    loads(y, model_result.model_id)
    assert (
        model_result.model_id.artefact_schema_id.as_string()
        == model_id.artefact_schema_id.as_string()
    )
    assert y.model == x.model
