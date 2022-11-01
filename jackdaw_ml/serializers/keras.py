import pyarrow as pa  # type: ignore
import tensorflow as tf

from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable

from typing import TypeVar, Optional

from jackdaw_ml.serializers.tensor import TensorSerializer

T = TypeVar("T")


class KerasSerializer(Serializable[tf.Variable]):
    @staticmethod
    def to_resource(item: tf.Variable) -> Resource:
        if not isinstance(item, tf.Variable):
            raise ValueError(f"Received {item}, expected {tf.Variable}")
        item_weight = pa.Tensor.from_numpy(item.numpy())
        return TensorSerializer.to_resource(item_weight)

    @staticmethod
    def from_resource(
        uninitialised_item: Optional[tf.Variable], buffer: Resource
    ) -> tf.Variable:
        item_weight: pa.Tensor = TensorSerializer.from_resource(None, buffer)
        if uninitialised_item is not None:
            uninitialised_item.assign(item_weight.to_numpy())
            return uninitialised_item
        return tf.Variable(item_weight.to_numpy())
