import logging
import os
from typing import *

import pytest

from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.serializers.pickle import PickleSerializer

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

T = TypeVar("T")


@pytest.fixture
def simple_model_example():
    @artefacts({PickleSerializer: "m"}, endpoint=remote_endpoint())
    class ModelExample:
        def __init__(self) -> None:
            self.m = 3

    return ModelExample


@pytest.fixture
def multiple_item_example():
    @artefacts({PickleSerializer: ["m", "n"]}, endpoint=remote_endpoint())
    class MultipleItem:
        def __init__(self) -> None:
            self.m = 3
            self.n = 4

    return MultipleItem


@pytest.fixture
def idempotent_model_example():
    @artefacts({PickleSerializer: "m"}, endpoint=remote_endpoint())
    @artefacts({PickleSerializer: "m"}, endpoint=remote_endpoint())
    @artefacts({PickleSerializer: "m"}, endpoint=remote_endpoint())
    @artefacts({PickleSerializer: "m"}, endpoint=remote_endpoint())
    @artefacts({PickleSerializer: "m"}, endpoint=remote_endpoint())
    class IdempotentModelExample:
        def __init__(self) -> None:
            self.m = 3

    return IdempotentModelExample


def remote_endpoint() -> ArtefactEndpoint:
    return ArtefactEndpoint.remote(TEST_API_KEY)


TEST_API_KEY = os.getenv("SHAREABLEAI_TEST_API_KEY", "Empty")
