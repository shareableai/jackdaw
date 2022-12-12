from jackdaw_ml.artefact_decorator import SupportsArtefacts


def trace_artefacts(cls: SupportsArtefacts, indent: int = 0):
    if not isinstance(cls, SupportsArtefacts):
        return None
    indentation = "\t" * (indent + 1)
    if len(cls.__artefact_slots__) == 0 and len(cls.__artefact_children__) == 0:
        print(f"{cls.__class__}" + "{}")
        return None
    if indent == 0:
        print(f"{cls.__class__}" + "{")
    for (artefact_name, serializer) in cls.__artefact_slots__.items():
        print(f"{indentation}({artefact_name}) [{serializer}]")
    for child_model in cls.__artefact_children__:
        print(
            f"{indentation}({child_model}) {getattr(cls, child_model).__class__}" + "{"
        )
        trace_artefacts(getattr(cls, child_model), indent + 1)
        print(f"{indentation}" + "}")
    if indent == 0:
        print("}")
