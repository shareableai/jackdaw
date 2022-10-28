import logging
import pathlib
import tempfile
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
from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable
from jackdaw_ml.vcs import get_current_hash

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


class ArtefactNotFound(AttributeError):
    """
    Expected to find an Artefact on the initialised model, but there wasn't anything in the slot
    """

    pass


def _get_artefact(cls, name: str) -> Any:
    storage_location: str = getattr(cls, "__storage_location__")
    if storage_location == "__dict__":
        return getattr(cls, name)
    else:
        return getattr(cls, storage_location).get(name)


def _set_artefact(cls, name: str, value) -> Any:
    storage_location: str = getattr(cls, "__storage_location__")
    if storage_location == "__dict__":
        return setattr(cls, name, value)
    else:
        return setattr(getattr(cls, storage_location), name, value)


def _has_artefact(cls, name: str) -> bool:
    try:
        return _get_artefact(cls, name) is not None
    except AttributeError:
        return False


def _has_children_or_artefacts(cls) -> bool:
    return (
                   hasattr(cls, "__artefact_slots__") and hasattr(cls, "__artefact_subclasses__")
           ) and (
                   len(getattr(cls, "__artefact_slots__")) > 0
                   or len(getattr(cls, "__artefact_subclasses__")) > 0
           )


def _list_artefacts(cls) -> Iterable[str]:
    """
    Iterate through all (potential) artefact items on a class
    """
    storage_location: str = getattr(cls, "__storage_location__")
    if storage_location == "__dict__":
        # Dict won't include all items, dir includes too many items
        for name in dir(cls):
            if not name.startswith("_"):
                yield name
    else:
        storage_dict: Dict[str, Any] = getattr(cls, storage_location)
        yield from storage_dict.keys()


def _load_artefact(
        self, artefact_name: str, serializer: Type[Serializable], artefact: Resource
):
    storage_location = getattr(self, "__storage_location__")
    if storage_location == "__dict__":
        target_location = self
    else:
        target_location = getattr(self, storage_location)
    setattr(
        target_location,
        artefact_name,
        serializer.from_resource(
            getattr(target_location, artefact_name),
            artefact,
        ),
    )


def _add_children(self) -> None:
    """
    Recursively add Model Children to Model Class

    This operation is carried out after objects are initialised during `dumps` and `loads`,
    which means that the items can contain a non-negligible amount of data within them.

    Class based operations are carried out within the call to `artefacts` on the recursive
    items.
    """
    if (
            not hasattr(self, "__artefact_subclasses__")
            or getattr(self, "__artefact_subclasses__") is None
    ):
        setattr(self, "__artefact_subclasses__", set())
    for artefact_name in _list_artefacts(self):
        for detector in getattr(self, "__detectors__"):
            if detector.is_child(_get_artefact(self, artefact_name)):
                if artefact_name not in getattr(
                        self, "__artefact_subclasses__"
                ) or not hasattr(_get_artefact(self, artefact_name), "loads"):
                    _set_artefact(
                        self,
                        artefact_name,
                        artefacts({}, metadata_slot_name=detector.storage_location)(
                            _get_artefact(self, artefact_name)
                        ),
                    )
                    # Recursively detect artefacts for newly created model child
                    LOGGER.info(
                        f"Setting Artefact with {detector.storage_location=} and {detector=}"
                    )
                    _detect_artefacts(_get_artefact(self, artefact_name))
                    _add_children(_get_artefact(self, artefact_name))
                    if _has_children_or_artefacts(_get_artefact(self, artefact_name)):
                        LOGGER.info(
                            f"Detected {artefact_name=} - {_get_artefact(self, artefact_name)=} as child using {detector}"
                        )
                        # Remove from tracking as Artefact, track as subclass
                        getattr(self, "__artefact_subclasses__").add(artefact_name)
                        if artefact_name in getattr(self, "__artefact_slots__"):
                            getattr(self, "__artefact_slots__").pop(artefact_name)


def _add_artefacts(
        cls,
        artefact_list: Dict[Type[Serializable], Union[List[str], str]],
        detectors: List[Detector] = None,
        storage_location: Optional[str] = None,
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
            hasattr(cls, "__artefact_subclasses__")
            and getattr(cls, "__artefact_subclasses__") is not None
    ):
        listed_artefact_classes = getattr(cls, "__artefact_subclasses__")
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
    setattr(cls, "__artefact_subclasses__", listed_artefact_classes)


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

    for item_name in _list_artefacts(cls):
        for detector in detectors:
            if (
                    detector.is_artefact(_get_artefact(cls, item_name))
                    and item_name not in listed_artefacts
            ):
                LOGGER.info(f"{item_name=} detected as artefact")
                listed_artefacts[item_name] = detector.serializer
    setattr(cls, "__artefact_slots__", listed_artefacts)


def _add_dumps(
        cls, endpoint: ArtefactEndpoint, auto_detect_artefacts: bool, model_name: str = None
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

    def dumps(self) -> PyModelID:
        LOGGER.info(f"Starting Dumps for {str(self)}")
        if auto_detect_artefacts:
            _detect_artefacts(self)
        for artefact_name in getattr(self, "__artefact_slots__"):
            if not hasattr(self, artefact_name):
                raise ArtefactNotFound
        _add_children(self)
        listed_artefacts: Dict[str, Type[Serializable]] = getattr(
            self, "__artefact_slots__"
        )
        child_ids = {
            child_name: getattr(self, child_name).dumps(getattr(self, child_name))
            for child_name in getattr(self, "__artefact_subclasses__")
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
                vcs_hash=get_current_hash().hash,
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
            f"Loading Model Data for {self} - {model_id.artefact_schema_id.as_hex_string()=} {model_id.artefact_schema_id.as_hex_string()=} {model_id.vcs_hash=} "
        )
        model_data = load_model_data(
            model_name=model_id.name,
            vcs_hash=model_id.vcs_hash,
            artefact_schema_id=model_id.artefact_schema_id,
            endpoint=endpoint.endpoint,
        )
        for child_name in getattr(self, "__artefact_subclasses__"):
            try:
                _get_artefact(self, child_name).loads(
                    _get_artefact(self, child_name),
                    model_data.child_id_by_slot(child_name),
                )
            except RuntimeError:
                pass
        for (artefact_name, serializer) in getattr(self, "__artefact_slots__").items():
            _load_artefact(
                self,
                artefact_name,
                serializer,
                Resource.from_artefact(model_data.artefact_by_slot(artefact_name)),
            )

    if not hasattr(cls, "loads"):
        setattr(cls, "loads", loads)


def _get_available_detectors() -> List[Detector]:
    all_imports = []
    try:
        from jackdaw_ml.detectors.torch import TorchDetector, TorchSeqDetector
        all_imports = all_imports + [TorchSeqDetector, TorchDetector]
    except (ImportError, NameError):
        pass
    try:
        from jackdaw_ml.detectors.torch_geo import TorchGeoSeqDetector
        all_imports = all_imports + [TorchGeoSeqDetector]
    except (ImportError, NameError):
        pass
    # TODO: Add LightGBM, XGBoost, etc.
    # TODO: Allow adding a detector 'set' by the user.
    return all_imports


def artefacts(
        artefact_serializers: Dict[Type[Serializable], Union[List[str], str]],
        detectors: List[Detector] = None,
        name: str = None,
        metadata_slot_name: str = None,
        endpoint: ArtefactEndpoint = ArtefactEndpoint.default(),
) -> Callable[[Type[T]], Type[T]]:
    """
    Add Artefact Save & Load to a Model

    Adds methods;
        * def dumps(self) - Saves the model, providing a Model ID
        * def loads(self, model_id: ModelID) - Loads the model given a Model ID

    :param endpoint: Target to save & load models - either local or remote
    :param metadata_slot_name: Slot on the class to store artefact metadata
    :param name: Name to be associated with the saved model
    :param detectors: Set of `Detectors` to identify child models
    :param artefact_serializers: Dictionary mapping Serializers to Artefacts, i.e. {SerializerA: ['slot_a', 'slot_b']}
    """
    LOGGER.info(f"Initializing Artefacts with {metadata_slot_name=} and {endpoint=}")
    if detectors is None:
        detectors = _get_available_detectors()
    if len(artefact_serializers) == 0:
        # If no artefacts are specifically listed, attempt auto-detection
        auto_detect_artefacts = True
    else:
        auto_detect_artefacts = False

    def artefact_decorator(cls: Type[T]) -> Type[T]:
        _add_artefacts(cls, artefact_serializers, detectors, metadata_slot_name)
        _add_dumps(cls, endpoint, auto_detect_artefacts, name)
        _add_loads(cls, endpoint, auto_detect_artefacts)
        return cls

    return artefact_decorator
