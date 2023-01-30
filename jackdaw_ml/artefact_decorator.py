from __future__ import annotations

__all__ = ["artefacts", "find_artefacts", "format_class_name"]

import logging
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, List, Type, TypeVar, Union
from uuid import uuid4

from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors.hook import DefaultDetectors
from jackdaw_ml.metric_logging import MetricLogger
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")

if TYPE_CHECKING:
    from jackdaw_ml.detectors import ArtefactDetector, ChildDetector


def format_class_name(class_name: str) -> str:
    return class_name.replace("<class '", "").replace("'>", "").split(".")[-1]


def _add_artefacts(
    cls,
    endpoint: ArtefactEndpoint,
    artefact_list: Dict[Type[Serializable], Union[List[str], str]],
    artefact_detectors: List[ArtefactDetector] = None,
    child_detectors: List[ChildDetector] = None,
    name: str = None,
) -> None:
    """
    Indicate Artefacts on a Class

    Arguments
    -----
    `artefact_list`
        Dictionary Mapping with Serializers as Keys, and the Items to be Serialized by that Serializer as items, i.e.
        {TorchSerializer: ['a', 'b']} would serialize items on slots 'a' and 'b' with TorchSerializer.

    `detectors`
        Set of `Detector` - classes that are used to identify if a parameter on `cls` should be considered as a child
        model. If a parameter is considered a child model, it will *not* be considered an artefact, although there
        can be artefacts on that newly identified child model.

    `storage_location`
        Location to expect Artefacts on this given `cls`. If not set, will expect to find artefacts on `__dict__`.

    Notes
    ------
    Adding Artefacts to a class should be idempotent - `_add_artefacts` can be called multiple times onto the same
    `cls` object with the same arguments, and cause the same result each time, with subsequent calls having no effect.
    """
    if artefact_detectors is None:
        artefact_detectors = list()
    if child_detectors is None:
        child_detectors = list()
    if name is None:
        name = format_class_name(str(cls))
    setattr(cls, "__model_name__", name)

    if (
        hasattr(cls, "__artefact_detectors__")
        and getattr(cls, "__artefact_detectors__") is not None
    ):
        artefact_detectors = artefact_detectors + getattr(cls, "__artefact_detectors__")
    if (
        hasattr(cls, "__child_detectors__")
        and getattr(cls, "__child_detectors__") is not None
    ):
        child_detectors = child_detectors + getattr(cls, "__child_detectors__")

    if (
        hasattr(cls, "__artefact_slots__")
        and getattr(cls, "__artefact_slots__") is not None
    ):
        listed_artefacts = getattr(cls, "__artefact_slots__")
    else:
        listed_artefacts = {}
    if (
        hasattr(cls, "__artefact_children__")
        and getattr(cls, "__artefact_children__") is not None
    ):
        listed_artefact_classes = getattr(cls, "__artefact_children__")
    else:
        listed_artefact_classes = dict()
    for (artefact_serializer, artefact_slots) in artefact_list.items():
        if isinstance(artefact_slots, list):
            for slot in artefact_slots:
                listed_artefacts[slot] = artefact_serializer
        else:
            listed_artefacts[artefact_slots] = artefact_serializer
    setattr(cls, "__child_detectors__", child_detectors)
    setattr(cls, "__artefact_detectors__", artefact_detectors)
    setattr(cls, "__artefact_slots__", listed_artefacts)
    setattr(cls, "__artefact_children__", listed_artefact_classes)
    setattr(cls, "__artefact_endpoint__", endpoint)


def _add_logger(cls, endpoint: ArtefactEndpoint) -> None:
    def _log_metric(self, metric_name: str, metric_value: float) -> None:
        if not hasattr(self, "_logger"):
            model_uuid = uuid4()
            setattr(self, "_model_uuid", model_uuid)
            setattr(
                cls,
                "_logger",
                MetricLogger(getattr(cls, "__model_name__"), model_uuid, endpoint),
            )
        with getattr(self, "_logger") as logger:
            logger.log(metric_name, metric_value)

    setattr(cls, "_log_metric", _log_metric)


def artefacts(
    artefact_serializers: Dict[Type[Serializable], Union[List[str], str]] = None,
    artefact_detectors: List[ArtefactDetector] = None,
    child_detectors: List[ChildDetector] = None,
    name: str = None,
    endpoint: ArtefactEndpoint = ArtefactEndpoint.default(),
) -> Callable[[T], T]:
    """
    Add Artefact Save & Load to a Model

    Adds methods;
        * def dumps(self) - Saves the model, providing a Model ID
        * def loads(self, model_id: ModelID) - Loads the model given a Model ID

    :param endpoint: Target to save & load models - either local or remote
    :param name: Name to be associated with the saved model
    :param artefact_serializers: Dictionary mapping Serializers to Artefacts, i.e. {SerializerA: ['slot_a', 'slot_b']}
    """
    LOGGER.info(f"Initializing Artefacts with {endpoint=}")
    if artefact_serializers is None:
        artefact_serializers = {}
    if artefact_detectors is None:
        artefact_detectors = list(DefaultDetectors.artefact_detectors().keys())
    if child_detectors is None:
        child_detectors = list(DefaultDetectors.child_detectors().keys())

    def artefact_decorator(cls: Type[T]) -> Type[T]:
        _add_artefacts(
            cls,
            endpoint,
            artefact_serializers,
            artefact_detectors,
            child_detectors,
            name=name,
        )
        _add_logger(cls, endpoint)
        return cls

    return artefact_decorator


# TODO: Change from being a partial to full function, as otherwise it's missing useful documentation.
find_artefacts = partial(
    artefacts, artefact_serializers={}, artefact_detectors=None, child_detectors=None
)
