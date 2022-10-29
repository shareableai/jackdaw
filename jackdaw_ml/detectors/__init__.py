from dataclasses import dataclass
from typing import Set, TypeVar, Type, Generic, Any, Optional, Union

from jackdaw_ml.detectors.class_detector import ChildModelDetector
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")


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
    elif origin is dict:
        key, value = typ.__args__
        if key is None and value is None:
            return True
        if not isinstance(obj, dict):
            return False
        assert isinstance(obj, dict)
        return all(is_type(k, key) and is_type(v, value) for (k, v) in obj.items())
    else:
        raise NotImplementedError


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
    storage_location: Optional[str] = None

    def is_child(self, item: Any) -> bool:
        for child_model_type in self.child_models:
            if isinstance(child_model_type, ChildModelDetector):
                if child_model_type(item):
                    return True
            else:
                if is_type(item, child_model_type):
                    return True
        return False

    def is_artefact(self, item: Any) -> bool:
        return any(isinstance(item, subtype) for subtype in self.artefact_types)

    def __hash__(self) -> int:
        return (
            hash(tuple(self.child_models))
            + hash(tuple(self.artefact_types))
            + hash(self.serializer)
            + hash(self.storage_location)
        )
