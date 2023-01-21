from typing import Optional

import numpy as np
import pyarrow as pa

from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable
from jackdaw_ml.serializers.tensor import TensorSerializer


class NumpySerializer(Serializable[np.ndarray]):
    @staticmethod
    def to_resource(item: np.ndarray) -> Resource:
        if not isinstance(item, np.ndarray):
            raise ValueError(f"Received {item}, expected {np.ndarray}")
        item_weight = pa.Tensor.from_numpy(item)
        return TensorSerializer.to_resource(item_weight)

    @staticmethod
    def from_resource(
        uninitialised_item: Optional[np.ndarray], buffer: Resource
    ) -> np.ndarray:
        item_weight: pa.Tensor = TensorSerializer.from_resource(None, buffer)
        return item_weight.to_numpy()
