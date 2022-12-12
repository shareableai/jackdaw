__all__ = ["TorchLightningDetector"]


from jackdaw_ml.detectors import Detector
from jackdaw_ml.access_interface import DictAccessInterface
from jackdaw_ml.detectors.hook import DefaultDetectors
from jackdaw_ml.serializers.tensor import TorchSerializer

import pytorch_lightning as pl
import torch.nn as nn

TorchLightningDetector = Detector(
    child_models={pl.LightningModule},
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
    access_interface=DictAccessInterface,
    storage_location="_modules",
)

DefaultDetectors.add_detector(TorchLightningDetector)
