from functools import lru_cache

import numpy as np
import torch

import logging

logging.getLogger().setLevel(logging.WARN)

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import TransformerModel

from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml import loads
from jackdaw_ml.detectors.torch import TorchDetector, TorchSeqDetector
from jackdaw_ml import saves

# As the class already exists, we can just add artefacts to it and rename it
TransformerModelWithArtefacts = artefacts(
    {}, child_detectors=[TorchSeqDetector, TorchDetector]
)(TransformerModel)


@lru_cache(maxsize=1)
def example_data() -> TimeSeries:
    return AirPassengersDataset().load()


@lru_cache(maxsize=1)
def similar_example_data() -> TimeSeries:
    return AirPassengersDataset().load()[0:64]


def np_float_equivalence(a: np.ndarray, b: np.ndarray) -> bool:
    # Are `a` and `b` within a reasonable distance of each other, accounting for internal machine error?
    diff = np.sum(a - b)
    return diff <= np.finfo(np.float32).eps


def assert_model_equivalence(
    model: torch.nn.Module, model_two: torch.nn.Module
) -> bool:
    for (key, value) in model.state_dict().items():
        if key not in model_two.state_dict():
            return False
        if not torch.equal(value, model_two.state_dict()[key]):
            return False
    return True


def test_darts_torch_equivalence():
    m1 = TransformerModelWithArtefacts(32, 32)
    m2 = TransformerModelWithArtefacts(32, 32)
    # DARTS' Transformer is a complex case for Jackdaw - while Jackdaw can find the correct slots to retrieve the
    # model from, DARTS only generates the model that contained those slots when data is provided, as the Transformer
    # model requires the data to exist to know the sizing for parameters.
    # It may be that there's an easier way to initialise models on load without saving them via Pickle, but my
    # experience with the library ends here.
    m1.fit(example_data(), epochs=1)
    model_id = saves(m1)
    # Due to the dynamic load, we can 'trick' DARTS into loading a model with similar starting values, then override
    #   that model with our own parameters.
    m2.fit(similar_example_data(), epochs=1)

    # Prove that our unloaded/tricked model isn't the same as our saved model (yet)
    assert not assert_model_equivalence(m1.model, m2.model)
    # Load in the pre-existing model's artefacts
    loads(m2, model_id)
    assert assert_model_equivalence(m1.model, m2.model)


if __name__ == "__main__":
    m1 = TransformerModelWithArtefacts(32, 32)
    m1.fit(example_data(), epochs=1)
    model_id = saves(m1)
