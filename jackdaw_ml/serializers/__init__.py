from abc import abstractmethod
from typing import Generic, TypeVar, Optional

from jackdaw_ml.resource import Resource

T = TypeVar("T")


class Serializable(Generic[T]):
    @staticmethod
    @abstractmethod
    def to_resource(item: T) -> Resource:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_resource(uninitialised_item: Optional[T], buffer: Resource) -> T:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.__class__)
