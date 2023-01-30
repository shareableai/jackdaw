__all__ = ["DictAccessInterface"]

import logging
from typing import Dict, TypeVar

from jackdaw_ml.access_interface import AccessInterface

C = TypeVar("C")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class DictAccessInterface(AccessInterface[Dict[str, T], T]):
    @staticmethod
    def _get_item(container: Dict[str, T], key: str) -> T:
        return container[key]

    @staticmethod
    def _set_item(container: Dict[str, T], key: str, value: T) -> None:
        container[key] = value

    @staticmethod
    def _items(container: Dict[str, T]) -> Dict[str, T]:
        return container

    @staticmethod
    def _from_dict(d: Dict[str, T]) -> Dict[str, T]:
        return d
