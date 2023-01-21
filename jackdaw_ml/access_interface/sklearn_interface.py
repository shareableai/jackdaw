__all__ = ["SkLearnInterface"]

import inspect
import logging
import numpy as np

from jackdaw_ml.access_interface import DefaultAccessInterface
from jackdaw_ml.detectors import ArtefactDetector

from typing import TypeVar, Dict, List, Set

from jackdaw_ml.serializers.numpy import NumpySerializer
from jackdaw_ml.serializers.pickle import PickleSerializer

C = TypeVar("C")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class SkLearnInterface(DefaultAccessInterface):
    @classmethod
    def _keys(cls, container: Dict[str, T]) -> List[str]:
        return [
            name
            for name in dir(container)
            # This is the internal rule for estimators in SKLearn - ends with a _ and doesn't start with dunder
            if name.endswith("_")
            and not name.startswith("__")
            and not inspect.ismethod(cls._get_item(container, name))
            and not inspect.isfunction(cls._get_item(container, name))
            # Warning: Technically we could have to save/load via a Property with a setter attribute.
            and not isinstance(getattr(type(container), name, None), property)
        ]

    @classmethod
    def additional_detectors(cls) -> Set[ArtefactDetector]:
        return {
            ArtefactDetector(
                artefact_types={int, float, str}, serializer=PickleSerializer
            ),
            ArtefactDetector(artefact_types={np.ndarray}, serializer=NumpySerializer),
        }

    @staticmethod
    def _items(container: Dict[str, T]) -> Dict[str, T]:
        return {
            k: v
            for (k, v) in container.__dict__.items()
            if k in SkLearnInterface._keys(container)
        }

    @staticmethod
    def _from_dict(d: Dict[str, T]) -> Dict[str, T]:
        return d
