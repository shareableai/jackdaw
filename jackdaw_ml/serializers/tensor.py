__all__ = ["TensorSerializer", "TorchSerializer"]


from typing import Optional, TypeVar

import pyarrow as pa  # type: ignore
import torch
from pyarrow import BufferOutputStream, BufferReader

from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")


class TensorSerializer(Serializable[pa.Tensor]):
    @staticmethod
    def to_resource(item: pa.Tensor) -> Resource:
        output_stream = BufferOutputStream()
        pa.ipc.write_tensor(item, output_stream)
        return Resource(output_stream.getvalue().to_pybytes())

    @staticmethod
    def from_resource(uninitialised_item: Optional[T], buffer: Resource) -> pa.Tensor:
        input_stream = BufferReader(buffer.__bytes__())
        return pa.ipc.read_tensor(input_stream)


class TorchSerializer(Serializable[torch.nn.Parameter]):
    @staticmethod
    def to_resource(item: torch.nn.Parameter) -> Resource:
        if not isinstance(item, torch.nn.Parameter):
            raise ValueError(f"Received {item}, expected {torch.nn.Parameter}")
        item_weight = pa.Tensor.from_numpy(item.detach().numpy())
        return TensorSerializer.to_resource(item_weight)

    @staticmethod
    def from_resource(
        uninitialised_item: Optional[torch.nn.Parameter], buffer: Resource
    ) -> torch.nn.Parameter:
        item_weight: pa.Tensor = TensorSerializer.from_resource(None, buffer)
        return torch.nn.Parameter(torch.from_numpy(item_weight.to_numpy()))
