import logging
import pathlib
import tempfile
from typing import Union, Dict, Type, TypeVar, List, Tuple, Any
from uuid import uuid4

from artefact_link import PyModelID, LocalArtefactPath, ModelData

from jackdaw_ml.access_interface import AccessInterface, DefaultAccessInterface
from jackdaw_ml.artefact_container import (
    SupportsArtefacts,
    _detect_children,
    _detect_artefacts,
)
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors import ArtefactDetector, ChildDetector
from jackdaw_ml.serializers import Serializable
from jackdaw_ml.trace import try_convert, sort_dict
from jackdaw_ml.vcs import get_vcs_info

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


def _saves(
    model_class: Union[SupportsArtefacts, Tuple[Any, AccessInterface]],
    endpoint: ArtefactEndpoint,
    artefact_detectors: List[ArtefactDetector],
    child_detectors: List[ChildDetector],
) -> PyModelID:
    if isinstance(model_class, SupportsArtefacts):
        access_interface = DefaultAccessInterface
        child_detectors = model_class.__child_detectors__
        artefact_detectors = model_class.__artefact_detectors__
        existing_artefacts: Dict[
            str, Type[Serializable]
        ] = model_class.__artefact_slots__
        model_children = _detect_children(
            model_class, child_detectors, artefact_detectors, endpoint
        )
        # At this point, artefact_children have been picked up.
        detected_artefacts = _detect_artefacts(
            model_class, set(model_children.keys()), artefact_detectors
        )
    elif isinstance(model_class, Tuple):
        model_children = _detect_children(
            model_class, child_detectors, artefact_detectors, endpoint
        )
        detected_artefacts = _detect_artefacts(
            model_class, set(model_children.keys()), artefact_detectors
        )
        (model_class, access_interface) = model_class
        existing_artefacts = dict()
    else:
        raise ValueError

    child_ids = {}
    for (child_name, child_interface) in model_children.items():
        child = access_interface.get_artefact(model_class, child_name)
        if (
            isinstance(child, SupportsArtefacts)
            and child_interface is DefaultAccessInterface
        ):
            child_ids[child_name] = _saves(
                child,
                child.__artefact_endpoint__,
                list(set(artefact_detectors) | child_interface.additional_detectors()),
                child_detectors,
            )
        else:
            child_ids[child_name] = _saves(
                (child, child_interface),
                endpoint,
                list(set(artefact_detectors) | child_interface.additional_detectors()),
                child_detectors,
            )

    with tempfile.TemporaryDirectory() as td:
        tempdir_path = pathlib.Path(td)
        local_artefact_files: List[LocalArtefactPath] = []
        for (artefact_name, serializer) in (
            sort_dict(detected_artefacts | existing_artefacts)
        ).items():
            item = access_interface.get_artefact(model_class, artefact_name)
            filename = tempdir_path / f"{uuid4()}.artefact"
            local_artefact_files.append(
                LocalArtefactPath(artefact_name, serializer.to_file(item, filename))
            )
        model = ModelData(
            name=str(model_class.__class__),
            vcs_info=get_vcs_info(),
            local_artefacts=local_artefact_files,
            children=child_ids,
        )
        return model.dumps(endpoint.endpoint, None)  # TODO: Add RunID if present


def saves(model_class: SupportsArtefacts) -> PyModelID:
    """Save a Jackdaw-Compatible Model

    Saving a model allows the model to be restored later from an initialised class. In very simple code, this should
    look like the following;

    ```python
    x = MyModel()
    model_id = jackdaw_ml.saves(x)
    y = MyModel()
    jackdaw_ml.loads(y, model_id)
    assert y == x
    ```

    Depending on the user setup, this will be saved locally (~/.artefact_storage by default), or remotely on ShareableAI
    cloud. Metadata around the model will also be saved - either to a SQLite database (~/.artefact_registry.sqlite by
    default) or remotely on ShareableAI Cloud.

    You can query the SQLite database yourself to see what's saved if you'd like, but there's also tools like
    [Corvus](https://github.com/shareableai/corvus) for a CLI, or the Search functionality within Jackdaw
    (jackdaw_ml.search) to search the models programmatically.

    Saving is performed by identifying the items on a class that are required for that model to function, and storing
    the item itself in storage, and information about that item in the database. More information on that is available
    in the [Jackdaw docs](https://github.com/shareableai/jackdaw/blob/main/docs/save.md)
    """
    if isinstance(model_class, SupportsArtefacts):
        for detector in model_class.__child_detectors__:
            if interface := detector.get_child_interface(model_class) is not None:
                return _saves(
                    (model_class, interface),
                    model_class.__artefact_endpoint__,
                    model_class.__artefact_detectors__,
                    model_class.__child_detectors__,
                )
        return _saves(
            model_class,
            model_class.__artefact_endpoint__,
            model_class.__artefact_detectors__,
            model_class.__child_detectors__,
        )
    else:
        raise ValueError(
            "Model Class provided must be initialised via @artefacts before calling loads or save"
        )
