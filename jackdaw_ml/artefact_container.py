from __future__ import annotations

__all__ = ["SupportsArtefacts"]

import logging
from typing import (TYPE_CHECKING, Any, Dict, List, Protocol, Set, Tuple, Type,
                    Union, runtime_checkable)

from jackdaw_ml.access_interface import AccessInterface, DefaultAccessInterface
from jackdaw_ml.artefact_decorator import _add_artefacts
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.serializers import Serializable

if TYPE_CHECKING:
    from jackdaw_ml.detectors import ArtefactDetector, ChildDetector

LOGGER = logging.getLogger(__name__)


class ArtefactNotFound(AttributeError):
    """
    Expected to find an Artefact on the initialised model, but there wasn't anything in the slot
    """

    pass


@runtime_checkable
class SupportsArtefacts(Protocol):
    """
    After running @artefacts, a class should also fulfil the Protocol SupportsArtefacts.
    """

    __artefact_endpoint__: ArtefactEndpoint
    __artefact_slots__: Dict[str, Type[Serializable]]
    __artefact_children__: Dict[str, "AccessInterface"]
    __child_detectors__: List[ChildDetector]
    __artefact_detectors__: List[ArtefactDetector]


def _detect_artefact_annotations(
    model_class: SupportsArtefacts,
    child_slots: Set[str],
    artefact_detectors: List[ArtefactDetector],
) -> Dict[str, Type[Serializable]]:
    artefacts: Dict[str, Type[Serializable]] = {}
    for (name, item_class) in getattr(model_class, "__annotations__", {}).items():
        if name not in child_slots:
            for detector in artefact_detectors:
                if detector.is_artefact_type(item_class):
                    artefacts[name] = detector.serializer
                    break
    return artefacts


def _detect_artefacts(
    model_class: Union[SupportsArtefacts, Tuple[Any, AccessInterface]],
    child_slots: Set[str],
    artefact_detectors: List[ArtefactDetector],
) -> Dict[str, Serializable]:
    if isinstance(model_class, SupportsArtefacts):
        access_interface = DefaultAccessInterface
        artefact_detectors = list(
            set(artefact_detectors) | set(model_class.__artefact_detectors__)
        )
    elif isinstance(model_class, Tuple):
        (model_class, access_interface) = model_class
    else:
        raise ValueError
    artefact_ids = {
        artefact_name: serializer
        for (artefact_name, _, serializer) in access_interface.list_artefacts(
            model_class, artefact_detectors
        )
        if artefact_name not in child_slots
    }
    if isinstance(model_class, SupportsArtefacts):
        model_class.__artefact_slots__ = artefact_ids
    return artefact_ids


def _detect_children(
    model_class: Union[SupportsArtefacts, Tuple[Any, Type[AccessInterface]]],
    child_detectors: List[ChildDetector],
    artefact_detectors: List[ArtefactDetector],
    endpoint: ArtefactEndpoint,
) -> Dict[str, Type[AccessInterface]]:
    if isinstance(model_class, SupportsArtefacts):
        access_interface = DefaultAccessInterface
        child_detectors = list(
            set(child_detectors) | set(model_class.__child_detectors__)
        )
        artefact_detectors = list(
            set(artefact_detectors) | set(model_class.__artefact_detectors__)
        )
    elif isinstance(model_class, Tuple):
        (model_class, access_interface) = model_class
    else:
        raise ValueError
    child_ids = dict(
        access_interface.list_children(model_class, child_detectors, artefact_detectors)
    )
    if isinstance(model_class, SupportsArtefacts):
        model_class.__artefact_children__ = child_ids
    for child_name, child_access_interface in child_ids.items():
        child = access_interface.get_artefact(model_class, child_name)
        try:
            _add_artefacts(
                child,
                endpoint,
                {},
                model_class.__artefact_detectors__,
                model_class.__child_detectors__,
            )
        except AttributeError:
            pass
        if child_access_interface is DefaultAccessInterface and isinstance(
            child, SupportsArtefacts
        ):
            _detect_children(child, child_detectors, artefact_detectors, endpoint)
        else:
            _detect_children(
                (child, child_access_interface),
                child_detectors,
                artefact_detectors,
                endpoint,
            )
    return child_ids
