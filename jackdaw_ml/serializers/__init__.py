__all__ = ["Serializable"]

import pathlib
from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from jackdaw_ml.resource import Resource

T = TypeVar("T")


class Serializable(Generic[T]):
    @staticmethod
    @abstractmethod
    def to_resource(item: T) -> Resource:
        raise NotImplementedError

    @classmethod
    def to_file(cls, item: T, filename: pathlib.Path) -> pathlib.Path:
        with open(filename, "wb") as f:
            f.write(cls.to_resource(item).inner)
        return filename

    @staticmethod
    @abstractmethod
    def from_resource(uninitialised_item: Optional[T], buffer: Resource) -> T:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.__class__)
