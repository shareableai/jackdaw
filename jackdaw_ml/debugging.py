from typing import Any


def artefact_debug(obj: Any) -> None:
    print(
        f"{obj.__artefact_endpoint__=}\n"
        f"{obj.__artefact_slots__=}\n"
        f"{obj.__artefact_children__=}\n"
        f"{obj.__access_interface__=}\n"
        f"{obj.__storage_location__=}\n"
        f"{obj.__detectors__=}"
    )
