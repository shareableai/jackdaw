__all__ = ["MetricLogger"]

import datetime
from typing import List, Tuple
from uuid import UUID

from artefact_link import PyModelRun

from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.run_id import RunID
from jackdaw_ml.vcs import get_vcs_info


class MetricQueue:
    """
    Queueing System to allow for remotely sending logs in batches, if required.

    By default, has a max_size of 1 - acting like no queue exists at all.
    """

    def __init__(
        self, endpoint: ArtefactEndpoint, model_run: PyModelRun, max_size: int = 1
    ):
        self.model_run = model_run
        self.endpoint: ArtefactEndpoint = endpoint
        self._inner: List[Tuple[datetime.datetime, str, float]] = list()
        self._max_size: int = max_size

    def flush(self) -> None:
        self.model_run.save_metrics(self.endpoint.endpoint, self._inner)
        self._inner = list()

    def log(self, metric_name: str, metric_value: float) -> None:
        self._inner.append((datetime.datetime.now(), metric_name, metric_value))
        if len(self._inner) >= self._max_size:
            self.flush()


class MetricLogger:
    def __init__(self, model_name: str, model_uuid: UUID, endpoint: ArtefactEndpoint):
        run_id = RunID.id(endpoint)
        vcs_info = get_vcs_info()
        model_run = PyModelRun(
            endpoint=endpoint.endpoint,
            run_id=run_id,
            model_uuid=str(model_uuid),
            model_name=model_name,
            vcs=vcs_info,
        )
        self.queue = MetricQueue(endpoint, model_run)

    def __enter__(self):
        return self.queue

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue.flush()
