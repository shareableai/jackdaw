from typing import Optional, TypeVar

import numpy as np
import pyarrow as pa  # type: ignore
import tensorflow as tf

from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable
from jackdaw_ml.serializers.tensor import TensorSerializer

T = TypeVar("T")


class KerasSerializer(Serializable[tf.Variable]):
    @staticmethod
    def to_resource(item: tf.Variable) -> Resource:
        if not isinstance(item, tf.Variable):
            raise ValueError(f"Received {item}, expected {tf.Variable}")

        item_ndarray = item.numpy()
        # `.numpy() can return a np.float value rather than a np.ndarray`
        if not isinstance(item_ndarray, np.ndarray):
            item_ndarray = np.array(item_ndarray)
        item_ndarray = pa.Tensor.from_numpy(item_ndarray)
        return TensorSerializer.to_resource(item_ndarray)

    @staticmethod
    def from_resource(
        uninitialised_item: Optional[tf.Variable], buffer: Resource
    ) -> tf.Variable:
        item_weight: pa.Tensor = TensorSerializer.from_resource(None, buffer)
        if uninitialised_item is not None:
            uninitialised_item.assign(item_weight.to_numpy())
            return uninitialised_item
        return tf.Variable(item_weight.to_numpy())
