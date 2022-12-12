__all__ = ['ListAccessInterface']

from abc import abstractmethod

from jackdaw_ml.access_interface import AccessInterface

import logging
from typing import TypeVar, Dict, List

C = TypeVar("C")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class ListAccessInterface(AccessInterface[List[T], T]):
    @classmethod
    @abstractmethod
    def get_index(cls, _container: List[T], key: str) -> int:
        return int(key)

    @classmethod
    def get_item(cls, container: List[T], key: str) -> T:
        return container[cls.get_index(container, key)]

    @classmethod
    @abstractmethod
    def get_item_name(cls, container_item: T, index: int) -> str:
        return str(index)

    @classmethod
    def set_item(cls, container: List[T], key: str, value: T) -> None:
        container[cls.get_index(container, key)] = value

    @classmethod
    def items(cls, container: List[T]) -> Dict[str, T]:
        return {
            cls.get_item_name(c, index): c
            for (index, c) in enumerate(container)
        }

    @staticmethod
    def from_dict(d: Dict[str, T]) -> List[T]:
        return list(d.values())