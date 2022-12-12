__all__ = ['AccessInterface', 'DefaultAccessInterface', 'DictAccessInterface']

import inspect
import logging
from abc import ABC
from typing import Generic, TypeVar, List, Dict, Optional

C = TypeVar("C")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class AccessInterface(ABC, Generic[C, T]):
    """
    Jackdaw relies upon named attributes to know where to put artefacts back.

    Some frameworks are less kind to our Jackdaw, and place artefacts in ordered
    lists, or other such complexities.

    AccessInterface abstracts this type of complexity, and allows Jackdaw to treat
    all types of containers as a simple dictionary.

    To get the items from the container, Jackdaw calls 'items'. To put items back,
    Jackdaw provides a dictionary of deserialized items to `from_dict`.
    It's up to the interface to restore the order they originally had.
    """

    @classmethod
    def keys(cls, container: Dict[str, T]) -> List[str]:
        return list(cls.items(container).keys())

    @staticmethod
    def get_item(container: C, key: str) -> T:
        raise NotImplementedError

    @staticmethod
    def set_item(container: C, key: str, value: T) -> None:
        raise NotImplementedError

    @staticmethod
    def items(container: C) -> Dict[str, T]:
        raise NotImplementedError

    @staticmethod
    def from_dict(d: Dict[str, T]) -> C:
        raise NotImplementedError


class DefaultAccessInterface(AccessInterface[Dict[str, T], T]):
    @classmethod
    def keys(cls, container: Dict[str, T]) -> List[str]:
        return [
            name
            for name in dir(container)
            if not name.startswith("_")
            and not inspect.ismethod(cls.get_item(container, name))
            and not inspect.isfunction(cls.get_item(container, name))
        ]

    @staticmethod
    def get_item(container: Dict[str, T], key: str) -> Optional[T]:
        # When accessing attributes, it's tricky to tell the difference between properties that will behave like
        #   functions, and actual attributes that contain values. Ignoring classes that happen to act like functions
        #   isn't viable either - i.e. Torch Modules are classes that have a __call__ property so that they can act like
        #   functions.
        try:
            item = getattr(container, key)
        except RuntimeError:
            logger.warning(f"Accessing {key} on {container} caused a runtime error")
            return None
        return item

    @staticmethod
    def set_item(container: Dict[str, T], key: str, value: T) -> None:
        setattr(container, key, value)

    @staticmethod
    def items(container: Dict[str, T]) -> Dict[str, T]:
        return container.__dict__

    @staticmethod
    def from_dict(d: Dict[str, T]) -> Dict[str, T]:
        return d


from jackdaw_ml.access_interface.dict_interface import DictAccessInterface
