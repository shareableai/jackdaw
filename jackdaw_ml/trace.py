from jackdaw_ml.artefact_decorator import SupportsArtefacts


def trace_artefacts(cls: SupportsArtefacts, indent: int = 0):
    if not isinstance(cls, SupportsArtefacts):
        raise ValueError("Class has not been initialised with artefacts")
    indentation = "\t" * (indent + 1)
    if indent == 0:
        print(f"{cls.__class__}" + "{")
    for (artefact_name, serializer) in cls.__artefact_slots__.items():
        print(f"{indentation}({artefact_name}) [{serializer}]")
    for child_model in cls.__artefact_children__:
        print(f"{indentation}({child_model}) {getattr(cls, child_model).__class__}" + "{")
        trace_artefacts(getattr(cls, child_model), indent + 1)
        print(f"{indentation}" + "}")
    if indent == 0:
        print("}")
