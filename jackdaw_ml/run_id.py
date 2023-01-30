from multiprocessing import current_process
from typing import Optional

from artefact_link import PyRunID

from jackdaw_ml.artefact_endpoint import ArtefactEndpoint


class RunID:
    _id: Optional[PyRunID] = None
    _pid: Optional[int] = current_process().pid

    @staticmethod
    def id(endpoint: ArtefactEndpoint) -> PyRunID:
        # If the process ID has changed, update the RunID
        if RunID._id is None or current_process().pid != RunID._pid:
            RunID._id = PyRunID(endpoint.endpoint)
        return RunID._id

    @staticmethod
    def reset_id() -> None:
        RunID._id = None


def start_run() -> None:
    RunID.reset_id()
