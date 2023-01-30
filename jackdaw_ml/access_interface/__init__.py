from __future__ import annotations

__all__ = ["AccessInterface", "DefaultAccessInterface", "DictAccessInterface"]

import inspect
import logging
from abc import ABC
from typing import (TYPE_CHECKING, Any, Dict, Generic, Iterable, List,
                    Optional, Tuple, Type, TypeVar, Union)

C = TypeVar("C")
T = TypeVar("T")

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from jackdaw_ml.artefact_container import SupportsArtefacts
    from jackdaw_ml.detectors import (ArtefactDetector, ChildDetector,
                                      Serializable)


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
    def _keys(cls, container: Dict[str, T]) -> List[str]:
        return list(cls._items(container).keys())

    @staticmethod
    def _get_item(container: C, key: str) -> T:
        raise NotImplementedError

    @staticmethod
    def _set_item(container: C, key: str, value: T) -> None:
        raise NotImplementedError

    @staticmethod
    def _items(container: C) -> Dict[str, T]:
        raise NotImplementedError

    @staticmethod
    def _from_dict(d: Dict[str, T]) -> C:
        raise NotImplementedError

    @classmethod
    def list_artefacts(
        cls,
        model: "SupportsArtefacts",
        artefact_detectors: "List[ArtefactDetector]",
    ) -> Iterable[Tuple[str, Any, "Serializable"]]:
        possible_artefact_names = cls._keys(model)
        artefact_slots = getattr(model, "__artefact_slots__", dict())
        for artefact_name in set(possible_artefact_names) - set(artefact_slots.keys()):
            for detector in artefact_detectors:
                if detector.is_artefact(cls.get_artefact(model, artefact_name)):
                    yield artefact_name, cls.get_artefact(
                        model, artefact_name
                    ), detector.serializer
                    break
        for artefact_name, serializer in artefact_slots.items():
            yield artefact_name, cls.get_artefact(model, artefact_name), serializer

    @classmethod
    def list_children(
        cls,
        model: "SupportsArtefacts",
        child_detectors: "List[ChildDetector]",
        artefact_detectors: "List[ArtefactDetector]",
    ) -> Iterable[Tuple[str, Type[AccessInterface]]]:
        identified_children = []
        for child_name in [a for a in cls._keys(model)]:
            if child_name is None or child_name == "__dict__":
                # This is self-referential - implying that the artefacts/children on the class are a child class.
                continue
            detectors: List[Union[ChildDetector, ArtefactDetector]] = [
                *child_detectors,
                *artefact_detectors,
            ]
            for detector in detectors:
                if (
                    child_interface := detector.get_child_interface(
                        cls.get_artefact(model, child_name)
                    )
                ) is not None:
                    identified_children.append(child_name)
                    yield child_name, child_interface
                    # Break detection loop - move to next child_name
                    break

    @classmethod
    def get_child(cls, model, child_name: str) -> "SupportsArtefacts":
        artefact = cls.get_artefact(model, child_name)
        if not isinstance(artefact, SupportsArtefacts):
            raise ValueError("Retrieved Artefact is not a Model")
        return artefact

    @classmethod
    def set_artefact(cls, model: "SupportsArtefacts", artefact_name: str, artefact: T):
        return cls._set_item(model, artefact_name, artefact)

    @classmethod
    def get_artefact(cls, model: "SupportsArtefacts", artefact_name: str) -> T:
        return cls._get_item(model, artefact_name)

    @classmethod
    def has_artefact(cls, model: "SupportsArtefacts", artefact_name: str) -> bool:
        try:
            return cls.get_artefact(model, artefact_name) is not None
        except AttributeError:
            return False


class DefaultAccessInterface(AccessInterface[Dict[str, T], T]):
    @classmethod
    def _keys(cls, container: Dict[str, T]) -> List[str]:
        return [
            name
            for name in dir(container)
            if not name.startswith("_")
            and not inspect.ismethod(cls._get_item(container, name))
            and not inspect.isfunction(cls._get_item(container, name))
        ]

    @staticmethod
    def _get_item(container: Dict[str, T], key: str) -> Optional[T]:
        # When accessing attributes, it's tricky to tell the difference between properties that will behave like
        #   functions, and actual attributes that contain values. Ignoring classes that happen to act like functions
        #   isn't viable either - i.e. Torch Modules are classes that have a __call__ property so that they can act like
        #   functions.
        try:
            item = getattr(container, key)
        except RuntimeError:
            logger.warning(f"Accessing {key} on {container} caused a runtime error")
            return None
        except AttributeError:
            return None
        return item

    @staticmethod
    def _set_item(container: Dict[str, T], key: str, value: T) -> None:
        setattr(container, key, value)

    @staticmethod
    def _items(container: Dict[str, T]) -> Dict[str, T]:
        return container.__dict__

    @staticmethod
    def _from_dict(d: Dict[str, T]) -> Dict[str, T]:
        return d


from jackdaw_ml.access_interface.dict_interface import DictAccessInterface
