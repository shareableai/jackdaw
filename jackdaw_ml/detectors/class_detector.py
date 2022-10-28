from abc import abstractmethod, ABCMeta
from typing import Type, Any


class ChildModuleDetector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, target_class: Type[Any]):
        raise NotImplementedError
