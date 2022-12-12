__all__ = ['DictAccessInterface']

from jackdaw_ml.access_interface import AccessInterface

import logging
from typing import TypeVar, Dict

C = TypeVar("C")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class DictAccessInterface(AccessInterface[Dict[str, T], T]):
    @staticmethod
    def get_item(container: Dict[str, T], key: str) -> T:
        return container[key]

    @staticmethod
    def set_item(container: Dict[str, T], key: str, value: T) -> None:
        container[key] = value

    @staticmethod
    def items(container: Dict[str, T]) -> Dict[str, T]:
        return container

    @staticmethod
    def from_dict(d: Dict[str, T]) -> Dict[str, T]:
        return d
