__all__ = ["PickleSerializer"]

import pickle
from typing import Optional, TypeVar

from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")


class PickleSerializer(Serializable[T]):
    @staticmethod
    def to_resource(item: T) -> Resource:
        return Resource(pickle.dumps(item))

    @staticmethod
    def from_resource(uninitialised_item: Optional[T], buffer: Resource) -> T:
        return pickle.loads(buffer.__bytes__())
