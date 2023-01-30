__all__ = ["Detector"]

from collections import OrderedDict
from dataclasses import dataclass
from typing import (Any, Dict, Generic, List, Optional, Set, Type, TypeVar,
                    Union)

from jackdaw_ml.access_interface.list_interface import ListAccessInterface
from jackdaw_ml.detectors.class_detector import ChildModelDetector
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")

from jackdaw_ml.access_interface import (AccessInterface,
                                         DefaultAccessInterface,
                                         DictAccessInterface)


def _get_origin(obj: object) -> Optional[object]:
    if hasattr(obj, "__origin__"):
        return obj.__origin__
    return None


def is_type(obj: object, typ: Type) -> bool:
    if typ is Any:
        return True
    origin = _get_origin(typ)
    if origin is None:
        return isinstance(obj, typ)
    elif origin is dict or origin is OrderedDict:
        key, value = typ.__args__
        if key is None and value is None:
            return True
        if not isinstance(obj, dict):
            return False
        return len(obj.items()) > 0 and all(
            is_type(k, key) and is_type(v, value) for (k, v) in obj.items()
        )
    elif origin is list or origin is List:
        if not (isinstance(obj, list) or isinstance(obj, List)):
            return False
        if len(obj) == 0:
            return False
        (list_type,) = typ.__args__
        return all(is_type(x, list_type) for x in obj)
    elif origin is set or origin is Set:
        if not (isinstance(obj, set) or isinstance(obj, Set)):
            return False
        if len(obj) == 0:
            return False
        (set_type,) = typ.__args__
        return all(is_type(x, set_type) for x in obj)
    else:
        raise NotImplementedError


@dataclass(slots=True)
class ChildDetector:
    child_models: Dict[Type[Any], Type[AccessInterface]]

    def get_child_interface(self, item: Any) -> Optional[Type[AccessInterface]]:
        for child_model_type, child_interface in self.child_models.items():
            if isinstance(child_model_type, ChildModelDetector):
                if child_model_type(item):
                    return DefaultAccessInterface
            elif is_type(item, child_model_type):
                return child_interface
        return None

    def __hash__(self) -> int:
        return hash(tuple(self.child_models))


@dataclass(slots=True)
class ArtefactDetector(Generic[T]):
    artefact_types: Set[Type[T]]
    serializer: Type[Serializable[T]]

    def is_artefact(self, item: Any) -> bool:
        return any(isinstance(item, subtype) for subtype in self.artefact_types)

    def is_artefact_type(self, item: Type[Any]) -> bool:
        return any((item == subtype) for subtype in self.artefact_types)

    def get_child_interface(self, item: Any) -> Optional[Type[AccessInterface]]:
        """Retrieve the AccessInterface for the item, if the item is an eligible child."""
        for subtype in self.artefact_types:
            if is_type(item, List[subtype]):
                return ListAccessInterface
            elif is_type(item, Set[subtype]):
                raise NotImplementedError
            elif is_type(item, Dict[str, subtype]):
                return DictAccessInterface
        return None

    def __hash__(self) -> int:
        return hash(tuple(self.artefact_types)) + hash(self.serializer)


@dataclass(slots=True)
class Detector(Generic[T]):
    """
    Generic Detector for Child Models and Artefacts that require a specific serializer

    Attributes
    ----------
    `child_models`
        Types or Classes to be detected as Child Models

    `artefact_types`
        Types or Classes to be detected as Artefacts

    `serializer`
        Class to Serialize an object from types in `artefact_types` to Bytes

    `storage_location`
        If set, identify the attribute on the module which `get` and `set` will use as a target.
        If not set, items will be set and retrieved from the `__dict__` attribute expected on
        all objects.
        If set and the object is of type `dict`, it will be replaced with an ArtefactDict
        to ensure it is possible to set methods on the class for `loads`/`dumps` etc.
    """

    child_models: Set[Union[Type[object], ChildModelDetector]]
    artefact_types: Set[Type[T]]
    serializer: Optional[Type[Serializable[T]]]
    access_interface: AccessInterface = DefaultAccessInterface()
    storage_location: Optional[str] = None

    def get_child_interface(self, item: Any) -> Optional[Type[AccessInterface]]:
        for child_model_type in self.child_models:
            if isinstance(child_model_type, ChildModelDetector):
                if child_model_type(item):
                    return DefaultAccessInterface
            elif is_type(item, child_model_type):
                return DefaultAccessInterface
        for subtype in self.artefact_types:
            if is_type(item, List[subtype]):
                return ListAccessInterface
            elif is_type(item, Set[subtype]):
                raise NotImplementedError
            elif is_type(item, Dict[str, subtype]):
                return DictAccessInterface
        return None

    def is_artefact(self, item: Any) -> bool:
        return any(isinstance(item, subtype) for subtype in self.artefact_types)

    def __hash__(self) -> int:
        return (
            hash(tuple(self.child_models))
            + hash(tuple(self.artefact_types))
            + hash(self.serializer)
            + hash(self.storage_location)
        )
