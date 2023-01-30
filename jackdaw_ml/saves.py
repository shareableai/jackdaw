import logging
import pathlib
import tempfile
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union
from uuid import uuid4

from artefact_link import LocalArtefactPath, ModelData, PyModelID

from jackdaw_ml.access_interface import AccessInterface, DefaultAccessInterface
from jackdaw_ml.artefact_container import (SupportsArtefacts,
                                           _detect_artefacts, _detect_children)
from jackdaw_ml.artefact_decorator import format_class_name
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors import ArtefactDetector, ChildDetector
from jackdaw_ml.serializers import Serializable
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
                child, child.__artefact_endpoint__, artefact_detectors, child_detectors
            )
        else:
            child_ids[child_name] = _saves(
                (child, child_interface), endpoint, artefact_detectors, child_detectors
            )

    with tempfile.TemporaryDirectory() as td:
        tempdir_path = pathlib.Path(td)
        local_artefact_files: List[LocalArtefactPath] = []
        for (artefact_name, serializer) in (
            detected_artefacts | existing_artefacts
        ).items():
            item = access_interface.get_artefact(model_class, artefact_name)
            filename = tempdir_path / f"{uuid4()}.artefact"
            local_artefact_files.append(
                LocalArtefactPath(artefact_name, serializer.to_file(item, filename))
            )
        model = ModelData(
            name=getattr(
                model_class, "__name__", format_class_name(str(model_class.__class__))
            ),
            vcs_info=get_vcs_info(),
            local_artefacts=local_artefact_files,
            children=child_ids,
        )
        return model.dumps(endpoint.endpoint, None)  # TODO: Add RunID if present


def saves(model_class: SupportsArtefacts) -> PyModelID:
    if isinstance(model_class, SupportsArtefacts):
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
