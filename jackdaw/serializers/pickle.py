import pickle
from typing import TypeVar, Optional

from jackdaw.resource import Resource
from jackdaw.serializers import Serializable

T = TypeVar("T")


class PickleSerializer(Serializable[T]):
    @staticmethod
    def to_resource(item: T) -> Resource:
        return Resource(pickle.dumps(item))

    @staticmethod
    def from_resource(uninitialised_item: Optional[T], buffer: Resource) -> T:
        return pickle.loads(buffer.__bytes__())
