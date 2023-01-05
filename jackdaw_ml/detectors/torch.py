__all__ = ["TorchSeqDetector", "TorchDetector"]

from typing import Dict, List, OrderedDict

from jackdaw_ml.access_interface.list_interface import ListAccessInterface
from jackdaw_ml.detectors import ChildDetector, ArtefactDetector
from jackdaw_ml.access_interface import DictAccessInterface, DefaultAccessInterface
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.tensor import TorchSerializer

import torch.nn as nn

TorchSeqDetector = ChildDetector(
    child_models={
        nn.ModuleList: ListAccessInterface,
        nn.Sequential: ListAccessInterface,
        Dict[str, nn.Module]: DictAccessInterface,
        OrderedDict[str, nn.Module]: DictAccessInterface,
        List[nn.Module]: ListAccessInterface,
        nn.Module: DefaultAccessInterface,
    },
)

TorchDetector = ArtefactDetector(
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
)

DefaultDetectors.add_detector(TorchSeqDetector, DetectionLevel.Specific)
DefaultDetectors.add_detector(TorchDetector)
