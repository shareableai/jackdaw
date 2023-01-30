import logging
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union

from artefact_link import PyModelID, load_model_data

from jackdaw_ml.access_interface import AccessInterface, DefaultAccessInterface
from jackdaw_ml.artefact_container import (SupportsArtefacts,
                                           _detect_artefact_annotations,
                                           _detect_artefacts, _detect_children)
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors import ArtefactDetector, ChildDetector, Detector
from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")


def _loads(
    model_class: Union[SupportsArtefacts, Tuple[Any, AccessInterface]],
    model_id: PyModelID,
    endpoint: ArtefactEndpoint,
    artefact_detectors: List[ArtefactDetector],
    child_detectors: List[ChildDetector],
) -> None:
    model_data = load_model_data(
        model_name=model_id.name,
        vcs_id=model_id.vcs_id,
        artefact_schema_id=model_id.artefact_schema_id,
        endpoint=endpoint.endpoint,
    )
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
        detected_artefacts = _detect_artefacts(
            model_class, set(model_children.keys()), artefact_detectors
        )
        detected_artefacts = detected_artefacts | _detect_artefact_annotations(
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

    for (artefact_name, serializer) in (
        detected_artefacts | existing_artefacts
    ).items():
        try:
            access_interface.set_artefact(
                model_class,
                artefact_name,
                serializer.from_resource(
                    uninitialised_item=access_interface.get_artefact(
                        model_class, artefact_name
                    ),
                    buffer=Resource.from_artefact(
                        model_data.artefact_by_slot(artefact_name)
                    ),
                ),
            )
        # TODO: Change from Runtime Error to custom missing artefact error
        except RuntimeError as e:
            LOGGER.error(f"Failed to Load '{artefact_name}': {e}")
            pass

    for (child_name, child_interface) in model_children.items():
        child = access_interface.get_artefact(model_class, child_name)
        child_model_id = model_data.child_id_by_slot(child_name)
        if (
            isinstance(child, SupportsArtefacts)
            and child_interface is DefaultAccessInterface
        ):
            _loads(
                child,
                child_model_id,
                child.__artefact_endpoint__,
                artefact_detectors,
                child_detectors,
            )
        else:
            _loads(
                (child, child_interface),
                child_model_id,
                endpoint,
                artefact_detectors,
                child_detectors,
            )


# TODO: Rename loads to load_model to make clearer from the loads module.
# TODO: Add typing to loads function
def loads(model_class: SupportsArtefacts, model_id: PyModelID) -> None:
    if isinstance(model_class, SupportsArtefacts):
        _loads(
            model_class,
            model_id,
            model_class.__artefact_endpoint__,
            model_class.__artefact_detectors__,
            model_class.__child_detectors__,
        )
    else:
        raise ValueError(
            "Model Class provided must be initialised via @artefacts before calling loads or save"
        )
