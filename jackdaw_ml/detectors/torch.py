__all__ = ["TorchSeqDetector", "TorchDetector"]


from jackdaw_ml.detectors import Detector
from jackdaw_ml.detectors.access_interface import DictAccessInterface
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.tensor import TorchSerializer

import torch.nn as nn

TorchSeqDetector = Detector(
    child_models={nn.Sequential, nn.ModuleList},
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
    access_interface=DictAccessInterface,
    storage_location="_modules",
)

TorchDetector = Detector(
    child_models={nn.Module},
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
)

DefaultDetectors.add_detector(TorchSeqDetector, DetectionLevel.Specific)
DefaultDetectors.add_detector(TorchDetector)
