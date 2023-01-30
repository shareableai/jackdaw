__all__ = ["ChildModelDetector"]

from abc import ABCMeta, abstractmethod
from typing import Any, Type


class ChildModelDetector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, target_class: Type[Any]):
        raise NotImplementedError
