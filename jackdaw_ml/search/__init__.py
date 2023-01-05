from __future__ import annotations

__all__ = ["Searcher"]

from artefact_link import (
    PyModelID,
    PyRunID,
    PyVcsInfo,
    search_for_models,
    PyMetricFilter,
    search_for_vcs_id,
    PyVcsID,
)

from enum import Enum
from typing import Union, List, Dict, Set, Optional
from dataclasses import dataclass

from jackdaw_ml.artefact_endpoint import ArtefactEndpoint


class Comparison(Enum):
    GT = ">"
    LT = "<"
    EQ = "="

    @staticmethod
    def from_str(string: str) -> Comparison:
        if string not in [v.value for v in Comparison.__members__.values()]:
            raise ValueError(f"Unknown comparison - {string}")
        return next(v for v in Comparison.__members__.values() if v.value == string)

    def __str__(self) -> str:
        match self:
            case Comparison.GT:
                return "greater"
            case Comparison.LT:
                return "less"
            case Comparison.EQ:
                return "equal"
            case _:
                raise RuntimeError("Unreachable")

    def __hash__(self) -> int:
        return hash(tuple([self.name, self.value]))


@dataclass(unsafe_hash=True)
class MetricFilter:
    metric_name: str
    metric_value: float
    ordering: Comparison


class Searcher:
    """
    Search for ShareableAI models over local and remote systems

    ```python
    # Finding matching models locally
    models = (Searcher(ArtefactEndpoint.default())
        .with_name(["xgboost", "tensorflow"])
        .with_metric("<", "accuracy", 0.5)
        .with_metric(">", "loss", 0.1)
        .models())

    # Finding Model Runs
    model_runs = (Searcher(ArtefactEndpoint.default())
        .with_name(["CarDrivingModel"])
        .with_vcs(VCS_ID)


    # Finding model metrics remotely
    model_metrics = (Searcher(ArtefactEndpoint.remote(MY_API_KEY))
        .with_name(["xgboost", "tensorflow"])
        .with_runs([RunIDOne])
        .with_metric("<", "accuracy", 0.5)
        .with_metric(">", "loss", 0.1)
        .metrics())

    # Finding Models by Repo
    Searcher(ArtefactEndpoint.default())
        .with_repository("https://github.com/shareableai/jackdaw")
        .models()
    ```
    """

    def __init__(self, endpoint: ArtefactEndpoint):
        self.endpoint = endpoint
        self.names: Set[str] = set()
        self.runs: Set[PyRunID] = set()
        self.metric_filters: Set[MetricFilter] = set()
        self.vcs_information: List[PyVcsInfo] = list()
        self.repository_name: Optional[str] = None

    def with_name(self, name: Union[str, List[str]]) -> Searcher:
        if isinstance(name, str):
            self.names.add(name)
        else:
            self.names = self.names | set(name)
        return self

    def with_runs(self, run: Union[PyRunID, Set[PyRunID]]) -> Searcher:
        if isinstance(run, str):
            self.runs.add(run)
        else:
            self.runs = self.runs | set(run)
        return self

    def with_repository(self, repository_name: str) -> Searcher:
        self.repository_name = repository_name
        return self

    def with_metric(
            self, comparison: Union[Comparison, str], name: str, value: float
    ) -> Searcher:
        if isinstance(comparison, str):
            comparison = Comparison.from_str(comparison)
        self.metric_filters.add(MetricFilter(name, value, comparison))
        return self

    def with_vcs(self, vcs_information: PyVcsInfo) -> Searcher:
        self.vcs_information.append(vcs_information)
        return self

    def _metric_filter(self) -> Optional[PyMetricFilter]:
        if len(self.metric_filters) == 0:
            return None
        initial_local_metric = self.metric_filters.pop()
        initial_metric = PyMetricFilter(
            initial_local_metric.metric_name,
            initial_local_metric.metric_value,
            str(initial_local_metric.ordering),
        )
        for metric in self.metric_filters:
            initial_metric = initial_metric.and_(metric)
        return initial_metric

    def models(self) -> Set[PyModelID]:
        """
        Return a unique set of Models that match the search criteria
        """
        if self.repository_name is not None:
            vcs_ids: List[PyVcsID] = search_for_vcs_id(
                self.endpoint.endpoint, self.repository_name
            )
        else:
            vcs_ids = list()
        return set(
            search_for_models(
                endpoint=self.endpoint.endpoint,
                names=list(self.names),
                runs=list(self.runs),
                metric_filter=self._metric_filter(),
                vcs_id=[x.id() for x in self.vcs_information] + vcs_ids,
            )
        )

    def metrics(self) -> Dict[str, Union[List[str], List[float]]]:
        """
        Return a Pandas-compatible dictionary of column names and attributes
        """
        raise NotImplementedError
