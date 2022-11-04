import torch
import torch.nn as nn

from jackdaw_ml.artefact_decorator import artefacts, SupportsArtefacts
from jackdaw_ml.debugging import artefact_debug
from jackdaw_ml.trace import trace_artefacts


@artefacts({})
class Model:
    def __init__(self):
        self.seq_model = nn.Sequential(nn.Linear(50, 2), nn.ReLU())


def test_sequential():
    x = Model()
    model_id = x.dumps()
    assert len(x.__artefact_children__) == 1
    assert isinstance(x.seq_model._modules["0"], SupportsArtefacts)
    assert isinstance(x.seq_model._modules["1"], SupportsArtefacts)
    assert len(x.seq_model._modules["1"].__artefact_children__) == 0
    assert len(x.seq_model._modules["1"].__artefact_slots__) == 0

    y = Model()
    y.loads(model_id)
    assert torch.equal(x.seq_model._modules["0"].bias, y.seq_model._modules["0"].bias)
