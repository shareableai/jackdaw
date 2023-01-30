from typing import List, OrderedDict

import torch
import torch.nn as nn

from jackdaw_ml import loads, saves
from jackdaw_ml.access_interface import (DefaultAccessInterface,
                                         DictAccessInterface)
from jackdaw_ml.access_interface.list_interface import ListAccessInterface
from jackdaw_ml.artefact_container import SupportsArtefacts
from jackdaw_ml.artefact_decorator import _add_artefacts, artefacts
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors import is_type
from jackdaw_ml.detectors.hook import DefaultDetectors


@artefacts({})
class Model:
    def __init__(self):
        self.seq_model = nn.Sequential(nn.Linear(50, 2), nn.ReLU())


def test_nn_sequential():
    x = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU())
    is_type(x._modules, List[nn.Sequential])


def test_detector():
    x = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU())
    _add_artefacts(x, ArtefactEndpoint.default(), {})
    res = dict(
        ListAccessInterface.list_children(
            x,
            list(DefaultDetectors.child_detectors().keys()),
            list(DefaultDetectors.artefact_detectors().keys()),
        )
    )
    assert is_type(x._modules, OrderedDict[str, nn.Module])
    assert len(dict(res).keys()) == 2


def test_sequential_dict():
    x = Model()
    _add_artefacts(x, ArtefactEndpoint.default(), {})
    res = ListAccessInterface.list_children(
        x.seq_model,
        list(DefaultDetectors.child_detectors().keys()),
        list(DefaultDetectors.artefact_detectors().keys()),
    )
    assert len(dict(res).keys()) == 2


@artefacts({})
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)


def test_module_detection():
    x = ConvNet()
    _add_artefacts(x, ArtefactEndpoint.default(), {})
    artefacts = {
        name
        for (name, _, _) in (
            DefaultAccessInterface.list_artefacts(
                x.conv1, list(DefaultDetectors.artefact_detectors().keys())
            )
        )
    }
    children = dict(
        DefaultAccessInterface.list_children(
            x,
            list(DefaultDetectors.child_detectors().keys()),
            list(DefaultDetectors.artefact_detectors().keys()),
        )
    )
    assert "weight" in artefacts
    assert "bias" in artefacts
    assert "conv1" in children
    assert "conv2" in children


def test_sequential():
    x = Model()
    model_id = saves(x)
    assert len(x.__artefact_children__) == 1
    assert isinstance(x.seq_model._modules["0"], SupportsArtefacts)
    assert isinstance(x.seq_model._modules["1"], SupportsArtefacts)
    assert len(x.seq_model._modules["1"].__artefact_children__) == 0
    assert len(x.seq_model._modules["1"].__artefact_slots__) == 0

    y = Model()
    loads(y, model_id)
    assert torch.equal(x.seq_model._modules["0"].bias, y.seq_model._modules["0"].bias)
