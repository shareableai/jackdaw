from __future__ import annotations

__all__ = ["artefacts", "SupportsArtefacts", "find_artefacts"]

import logging
import inspect
import pathlib
import tempfile

from functools import partial
from typing import (
    Type,
    List,
    Dict,
    Union,
    Callable,
    TypeVar,
    Any,
    Optional,
    Iterable,
    Protocol,
    Set,
    runtime_checkable,
)
from uuid import uuid4

from artefact_link import (
    ModelData,
    PyModelID,
    load_model_data,
    LocalArtefactPath,
)

from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors import Detector
from jackdaw_ml.access_interface import (
    AccessInterface,
    DefaultAccessInterface,
)
from jackdaw_ml.detectors.hook import DefaultDetectors
from jackdaw_ml.metric_logging import MetricLogger
from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable
from jackdaw_ml.vcs import get_vcs_info

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


class ArtefactNotFound(AttributeError):
    """
    Expected to find an Artefact on the initialised model, but there wasn't anything in the slot
    """

    pass


@runtime_checkable
class SupportsArtefacts(Protocol):
    __artefact_endpoint__: ArtefactEndpoint
    __artefact_slots__: Dict[str, Type[Serializable]]
    __artefact_children__: Set[str]
    __access_interface__: Type[AccessInterface]
    __storage_location__: Optional[str]
    __detectors__: List[Detector]


def _get_artefact(cls: SupportsArtefacts, name: str) -> Any:
    access_interface: Type[AccessInterface] = cls.__access_interface__
    storage_location: Optional[str] = cls.__storage_location__
    if storage_location is None or storage_location == "__dict__":
        return access_interface.get_item(cls, name)
    else:
        return access_interface.get_item(getattr(cls, storage_location), name)


def _set_artefact(cls: SupportsArtefacts, name: str, value) -> None:
    access_interface: Type[AccessInterface] = cls.__access_interface__
    storage_location: Optional[str] = cls.__storage_location__
    if storage_location is None or storage_location == "__dict__":
        return access_interface.set_item(cls, name, value)
    else:
        return access_interface.set_item(getattr(cls, storage_location), name, value)


def _has_artefact(cls, name: str) -> bool:
    try:
        return _get_artefact(cls, name) is not None
    except AttributeError:
        return False


def _list_potential_artefacts(cls: SupportsArtefacts) -> Iterable[str]:
    """
    Iterate through all (potential) artefact items on a class
    """
    storage_location: str = cls.__storage_location__
    access_interface: Type[AccessInterface] = cls.__access_interface__
    if storage_location == "__dict__" or storage_location is None:
        return access_interface.keys(cls)  # type: ignore
    else:
        return access_interface.keys(getattr(cls, storage_location))


def _load_artefact(
    self, artefact_name: str, serializer: Type[Serializable], artefact: Resource
):
    _set_artefact(
        self,
        artefact_name,
        serializer.from_resource(
            _get_artefact(self, artefact_name)
            if _has_artefact(self, artefact_name)
            else None,
            artefact,
        ),
    )


def _add_children(self: SupportsArtefacts) -> None:
    """
    Recursively add Model Children to Model Class

    This operation is carried out after objects are initialised during `dumps` and `loads`,
    which means that the items can contain a non-negligible amount of data within them.

    Class based operations are carried out within the call to `artefacts` on the recursive
    items.
    """
    if (
        not hasattr(self, "__artefact_children__")
        or getattr(self, "__artefact_children__") is None
    ):
        self.__artefact_children__ = set()
    for artefact_name in _list_potential_artefacts(self):
        for detector in self.__detectors__:
            if detector.is_child(_get_artefact(self, artefact_name)):
                if artefact_name not in self.__artefact_children__ or not hasattr(
                    _get_artefact(self, artefact_name), "loads"
                ):
                    _set_artefact(
                        self,
                        artefact_name,
                        artefacts(
                            {},
                            detectors=self.__detectors__,
                            metadata_slot_name=detector.storage_location,
                            access_interface=detector.access_interface,
                        )(_get_artefact(self, artefact_name)),
                    )
                    # Recursively detect artefacts for newly created model child
                    LOGGER.info(
                        f"Setting Artefact with {detector.storage_location=} and {detector=}"
                    )
                    _detect_artefacts(_get_artefact(self, artefact_name))
                    _add_children(_get_artefact(self, artefact_name))
                    # New Model Child isn't detected as a child fully until it's established that the child has
                    #   artefacts or children of its own, otherwise it's meaningless to track it.
                    potential_child = _get_artefact(self, artefact_name)
                    # if str(potential_child.__class__) == "<class 'torch.nn.modules.transformer.TransformerDecoder'>":
                    #    breakpoint()
                    if isinstance(potential_child, SupportsArtefacts) and (
                        len(potential_child.__artefact_children__) > 0
                        or len(potential_child.__artefact_slots__) > 0
                    ):
                        LOGGER.info(
                            f"Detected {artefact_name=} - {potential_child=} as child using {detector}"
                        )
                        # Remove from tracking as Artefact, track as subclass
                        self.__artefact_children__.add(artefact_name)
                        if artefact_name in self.__artefact_slots__:
                            self.__artefact_slots__.pop(artefact_name)


def _add_artefacts(
    cls,
    artefact_list: Dict[Type[Serializable], Union[List[str], str]],
    detectors: List[Detector] = None,
    storage_location: Optional[str] = None,
    access_interface: Type[AccessInterface] = DefaultAccessInterface,
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
    if detectors is None:
        detectors = list()
    if storage_location is None:
        storage_location = "__dict__"
    if (
        not hasattr(cls, "__storage_location__")
        or getattr(cls, "__storage_location__") is None
    ):
        setattr(cls, "__storage_location__", storage_location)
    if (
        not hasattr(cls, "__access_interface__")
        or getattr(cls, "__access_interface__") is None
    ):
        setattr(cls, "__access_interface__", access_interface)
    if hasattr(cls, "__detectors__") and getattr(cls, "__detectors__") is not None:
        detectors = detectors + getattr(cls, "__detectors__")
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
        listed_artefact_classes = set()
    for (artefact_serializer, artefact_slots) in artefact_list.items():
        if isinstance(artefact_slots, list):
            for slot in artefact_slots:
                listed_artefacts[slot] = artefact_serializer
        else:
            listed_artefacts[artefact_slots] = artefact_serializer
    setattr(cls, "__detectors__", detectors)
    setattr(cls, "__artefact_slots__", listed_artefacts)
    setattr(cls, "__artefact_children__", listed_artefact_classes)


def _detect_artefacts(cls) -> None:
    if not hasattr(cls, "__detectors__"):
        return None
    detectors = getattr(cls, "__detectors__")
    if not hasattr(cls, "__artefact_slots__") or (
        getattr(cls, "__artefact_slots__") is None
    ):
        listed_artefacts: Dict[str, Type[Serializable]] = {}
    else:
        listed_artefacts = getattr(cls, "__artefact_slots__")

    for item_name in _list_potential_artefacts(cls):
        for detector in detectors:
            if (
                detector.is_artefact(_get_artefact(cls, item_name))
                and item_name not in listed_artefacts
            ):
                LOGGER.info(f"{item_name=} detected as artefact")
                listed_artefacts[item_name] = detector.serializer
    setattr(cls, "__artefact_slots__", listed_artefacts)


def _add_logger(cls, endpoint: ArtefactEndpoint) -> None:
    def _log_metric(self, metric_name: str, metric_value: float) -> None:
        if not hasattr(self, "_logger"):
            model_uuid = uuid4()
            setattr(self, "_model_uuid", model_uuid)
            setattr(cls, "_logger", MetricLogger(getattr(cls, "__model_name__"), model_uuid, endpoint))
        with getattr(self, "_logger") as logger:
            logger.log(metric_name, metric_value)

    setattr(cls, "_log_metric", _log_metric)


def _add_dumps(
    cls,
    endpoint: ArtefactEndpoint,
    auto_detect_artefacts: bool,
    model_name: str = None,
) -> None:
    """Add a `dumps` function to the class that provides a way to save the
    model artefacts to the indicated `store`.

    :param cls:
    :param endpoint: Target for Saving Model
    :param auto_detect_artefacts: Whether to detect child artefacts automatically
    :param model_name: Name of Model
    :return:
    """
    setattr(cls, "__artefact_endpoint__", endpoint)
    if model_name is None:
        model_name = str(cls)
    setattr(cls, "__model_name__", model_name)

    def dumps(self) -> PyModelID:
        LOGGER.info(f"Starting Dumps for {str(self)}")
        if auto_detect_artefacts:
            _detect_artefacts(self)
        for artefact_name in getattr(self, "__artefact_slots__"):
            if not _has_artefact(self, artefact_name):
                raise ArtefactNotFound
        _add_children(self)
        listed_artefacts: Dict[str, Type[Serializable]] = getattr(
            self, "__artefact_slots__"
        )
        child_ids = {
            child_name: _get_artefact(self, child_name).dumps()
            if inspect.ismethod(_get_artefact(self, child_name).dumps)
            else _get_artefact(self, child_name).dumps(_get_artefact(self, child_name))
            for child_name in getattr(self, "__artefact_children__")
        }

        # Loading all Artefacts into memory will cause RAM usage to double.
        #   Saving them into temporary files allows processes to read over
        #   them multiple times without having to keep them in memory.
        with tempfile.TemporaryDirectory() as td:
            tempdir_path = pathlib.Path(td)
            local_artefact_files: List[LocalArtefactPath] = []
            for (item, serializer) in listed_artefacts.items():
                if not _has_artefact(self, item):
                    raise ArtefactNotFound(f"{self} has no attribute {item}")
                filename = tempdir_path / f"{uuid4()}.artefact"
                local_artefact_files.append(
                    LocalArtefactPath(path=str(filename.absolute()), slot=item)
                )
                res = serializer.to_resource(_get_artefact(self, item))
                LOGGER.info(f"Saving {model_name= } - {item=} - {hash(res)=}")
                with open(filename, "wb") as target_file:
                    target_file.write(res.inner)

            model = ModelData(
                name=model_name if model_name is not None else str(self),
                vcs_info=get_vcs_info(),
                local_artefacts=local_artefact_files,
                children=child_ids,
            )
            return model.dumps(endpoint.endpoint)

    if not hasattr(cls, "dumps"):
        setattr(cls, "dumps", dumps)


def _add_loads(cls, endpoint: ArtefactEndpoint, auto_detect_artefacts: bool) -> None:
    setattr(cls, "__artefact_endpoint__", endpoint)

    def loads(self, model_id: PyModelID) -> None:
        if auto_detect_artefacts:
            _detect_artefacts(self)
        _add_children(self)
        LOGGER.info(
            f"Loading Model Data for {self} - {model_id.artefact_schema_id.as_hex_string()=} {model_id.artefact_schema_id.as_hex_string()=} {model_id.vcs_id=} "
        )
        model_data = load_model_data(
            model_name=model_id.name,
            vcs_id=model_id.vcs_id,
            artefact_schema_id=model_id.artefact_schema_id,
            endpoint=endpoint.endpoint,
        )
        for (artefact_name, serializer) in getattr(self, "__artefact_slots__").items():
            _load_artefact(
                self,
                artefact_name,
                serializer,
                Resource.from_artefact(model_data.artefact_by_slot(artefact_name)),
            )
        for child_name in getattr(self, "__artefact_children__"):
            try:
                if inspect.ismethod(_get_artefact(self, child_name).loads):
                    _get_artefact(self, child_name).loads(
                        model_data.child_id_by_slot(child_name),
                    )
                else:
                    _get_artefact(self, child_name).loads(
                        _get_artefact(self, child_name),
                        model_data.child_id_by_slot(child_name),
                    )
            except RuntimeError:
                pass

    if not hasattr(cls, "loads"):
        setattr(cls, "loads", loads)


def artefacts(
    artefact_serializers: Dict[Type[Serializable], Union[List[str], str]] = None,
    detectors: List[Detector] = None,
    name: str = None,
    metadata_slot_name: str = None,
    access_interface: Type[AccessInterface] = DefaultAccessInterface,
    endpoint: ArtefactEndpoint = ArtefactEndpoint.default(),
) -> Callable[[Type[T]], Type[T]]:
    """
    Add Artefact Save & Load to a Model

    Adds methods;
        * def dumps(self) - Saves the model, providing a Model ID
        * def loads(self, model_id: ModelID) - Loads the model given a Model ID

    :param endpoint: Target to save & load models - either local or remote
    :param metadata_slot_name: Slot on the class to store artefact metadata
    :param access_interface: Method by which to access artefacts on the class
    :param name: Name to be associated with the saved model
    :param detectors: Set of `Detectors` to identify child models
    :param artefact_serializers: Dictionary mapping Serializers to Artefacts, i.e. {SerializerA: ['slot_a', 'slot_b']}
    """
    LOGGER.info(f"Initializing Artefacts with {metadata_slot_name=} and {endpoint=}")
    if artefact_serializers is None:
        artefact_serializers = {}
    if detectors is None:
        detectors = list(DefaultDetectors.detectors().keys())
    if len(artefact_serializers) == 0:
        # If no artefacts are specifically listed, attempt auto-detection
        auto_detect_artefacts = True
    else:
        auto_detect_artefacts = False

    def artefact_decorator(cls: Type[T]) -> Type[T]:
        _add_artefacts(
            cls, artefact_serializers, detectors, metadata_slot_name, access_interface
        )
        _add_dumps(cls, endpoint, auto_detect_artefacts, model_name=name)
        _add_logger(cls, endpoint)
        _add_loads(cls, endpoint, auto_detect_artefacts)
        return cls

    return artefact_decorator


find_artefacts = partial(
    artefacts, artefact_serializers={}, detectors=DefaultDetectors.detectors().keys()
)
