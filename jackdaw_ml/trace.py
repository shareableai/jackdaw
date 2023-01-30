from typing import Any, Dict, Tuple, Type, Union

from jackdaw_ml.access_interface import AccessInterface, DefaultAccessInterface
from jackdaw_ml.artefact_container import (SupportsArtefacts,
                                           _detect_artefact_annotations,
                                           _detect_artefacts, _detect_children)
from jackdaw_ml.detectors.hook import DefaultDetectors
from jackdaw_ml.serializers import Serializable


def trace_artefacts(model_class: SupportsArtefacts):
    artefact_detectors = list(DefaultDetectors.artefact_detectors().keys())
    child_detectors = list(DefaultDetectors.child_detectors().keys())
    _trace_artefacts(
        model_class,
        artefact_detectors=artefact_detectors,
        child_detectors=child_detectors,
        indent=0,
    )


def _trace_artefacts(
    model_class: Union[SupportsArtefacts, Tuple[Any, AccessInterface]],
    artefact_detectors,
    child_detectors,
    indent: int = 0,
):
    if isinstance(model_class, SupportsArtefacts):
        access_interface = DefaultAccessInterface
        child_detectors = model_class.__child_detectors__
        artefact_detectors = model_class.__artefact_detectors__
        existing_artefacts: Dict[
            str, Type[Serializable]
        ] = model_class.__artefact_slots__
        model_children = _detect_children(
            model_class, child_detectors, artefact_detectors, None
        )
        detected_artefacts = _detect_artefacts(
            model_class, set(model_children.keys()), artefact_detectors
        )
        detected_artefacts = detected_artefacts | _detect_artefact_annotations(
            model_class, set(model_children.keys()), artefact_detectors
        )
    elif isinstance(model_class, Tuple):
        model_children = _detect_children(
            model_class, child_detectors, artefact_detectors, None
        )
        detected_artefacts = _detect_artefacts(
            model_class, set(model_children.keys()), artefact_detectors
        )
        (model_class, access_interface) = model_class
        existing_artefacts = dict()
    else:
        raise ValueError

    indentation = "\t" * (indent + 1)
    if (
        len((detected_artefacts | existing_artefacts).keys()) == 0
        and len(model_children.keys()) == 0
    ):
        print(f"{model_class.__class__}" + "{}")
        return None
    if indent == 0:
        print(f"{model_class.__class__}" + "{")
    for (artefact_name, serializer) in (
        detected_artefacts | existing_artefacts
    ).items():
        print(f"{indentation}({artefact_name}) [{serializer}]")
    for (child_model_name, child_model_interface) in model_children.items():
        print(f"{indentation}({child_model_name})" + "{")
        child = access_interface.get_artefact(model_class, child_model_name)
        if (
            isinstance(child, SupportsArtefacts)
            and child_model_interface is DefaultAccessInterface
        ):
            _trace_artefacts(child, artefact_detectors, child_detectors, indent + 1)
        else:
            _trace_artefacts(
                (child, child_model_interface),
                artefact_detectors,
                child_detectors,
                indent + 1,
            )
        print(f"{indentation}" + "}")
    if indent == 0:
        print("}")
