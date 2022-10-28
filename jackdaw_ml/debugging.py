from typing import Any


def artefact_debug(obj: Any) -> None:
    print(
        f"{obj.__artefact_endpoint__=}"
        f"{obj.__artefact_slots__=}"
        f"{obj.__artefact_subclasses__}"
    )
