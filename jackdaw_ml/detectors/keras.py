__all__ = ["KerasSeqDetector", "KerasDetector"]

import keras
import tensorflow as tf

from jackdaw_ml.detectors import Detector
from jackdaw_ml.detectors.access_interface import AccessInterface
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.keras import KerasSerializer

from typing import List, Dict


# If List patterns occur frequently, create a generic ListAccessInterface ABC with
#   abstract method `get_index` and `get_name`
class KerasLayerAccessInterface(AccessInterface[List[tf.Variable], tf.Variable]):
    @staticmethod
    def get_index(container: List[tf.Variable], key: str) -> int:
        index = next(
            iter([i for (i, c) in enumerate(container) if f"{KerasLayerAccessInterface.get_name(c, i)}" == key]), None
        )
        if index is None:
            raise KeyError(f"Could not find {key} on container")
        return index

    @staticmethod
    def get_name(container_item: tf.Variable, index: int) -> str:
        segments: List[str] = container_item.name.split('/', 2)
        name_component = segments[0].split('_')[0]
        if len(segments) > 1:
            return f"{name_component}_{index}/{segments[1]}"
        else:
            return f"{name_component}_{index}"

    @staticmethod
    def get_item(container: List[tf.Variable], key: str) -> tf.Variable:
        return container[KerasLayerAccessInterface.get_index(container, key)]

    @staticmethod
    def set_item(container: List[tf.Variable], key: str, value: tf.Variable) -> None:
        container[KerasLayerAccessInterface.get_index(container, key)] = value

    @staticmethod
    def items(container: List[tf.Variable]) -> Dict[str, tf.Variable]:
        return {KerasLayerAccessInterface.get_name(c, index): c for (index, c) in enumerate(container)}

    @staticmethod
    def from_dict(d: Dict[str, tf.Variable]) -> List[tf.Variable]:
        return list(d.values())


KerasSeqDetector = Detector(
    child_models={keras.Sequential},
    artefact_types=set(),
    serializer=None,
    access_interface=KerasLayerAccessInterface,
    storage_location="layers",
)

KerasDetector = Detector(
    child_models={keras.engine.base_layer.Layer},
    artefact_types={tf.Variable},
    serializer=KerasSerializer,
    access_interface=KerasLayerAccessInterface,
    storage_location="weights",
)

DefaultDetectors.add_detector(KerasSeqDetector, DetectionLevel.Specific)
DefaultDetectors.add_detector(KerasDetector, DetectionLevel.Generic)
