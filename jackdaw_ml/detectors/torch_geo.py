from typing import Any, Type

from jackdaw_ml.detectors import Detector, ChildModuleDetector
from jackdaw_ml.serializers.tensor import TorchSerializer

import torch_geometric.nn as nn_geo
import torch.nn as nn


class GeoSequentialDetector(ChildModuleDetector):
    """Sequential in Torch Geometric is defined as a subclass on Torch Sequential,
    but generated dynamically via Jinja. This additional complexity means that detecting
    it as a Torch module that happens to be called 'Sequential *something* is the easiest route.
    """

    def __call__(self, item: Type[Any]):
        return (
                isinstance(item, nn.Module)
                and next(iter(str(item.__class__).split("_", 2))) == "<class 'Sequential"
        )


TorchGeoSeqDetector = Detector(
    child_modules={GeoSequentialDetector()},
    artefact_types={nn.Parameter},
    serializer=TorchSerializer,
    storage_location="_modules",
)
