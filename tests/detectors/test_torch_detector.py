import torch
import torch.nn as nn

from jackdaw_ml.artefact_decorator import artefacts, _has_children_or_artefacts


@artefacts({})
class Model:
    def __init__(self):
        self.seq_model = nn.Sequential(nn.Linear(50, 2), nn.ReLU())


def test_sequential():
    x = Model()
    model_id = x.dumps()
    assert len(x.__artefact_subclasses__) == 1
    assert _has_children_or_artefacts(x.seq_model._modules["0"])
    assert not _has_children_or_artefacts(x.seq_model._modules["1"])

    y = Model()
    y.loads(model_id)
    assert torch.equal(x.seq_model._modules["0"].bias, y.seq_model._modules["0"].bias)
