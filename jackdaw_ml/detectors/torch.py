from jackdaw_ml.detectors import Detector
from jackdaw_ml.serializers.tensor import TorchSerializer

import torch.nn as nn

TorchSeqDetector = Detector(
    child_modules={nn.Sequential, nn.ModuleList},
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
    storage_location="_modules",
)

TorchDetector = Detector(
    child_modules={nn.Module},
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
)
