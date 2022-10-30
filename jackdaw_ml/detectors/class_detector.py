__all__ = ["ChildModelDetector"]

from abc import abstractmethod, ABCMeta
from typing import Type, Any


class ChildModelDetector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, target_class: Type[Any]):
        raise NotImplementedError
