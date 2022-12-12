__all__ = ["KerasSeqDetector", "KerasDetector"]

import keras
import tensorflow as tf

from jackdaw_ml.access_interface.list_interface import ListAccessInterface
from jackdaw_ml.detectors import Detector
from jackdaw_ml.access_interface import AccessInterface
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.keras import KerasSerializer

from typing import List, Dict


class KerasLayerAccessInterface(ListAccessInterface[AccessInterface[List[tf.Variable], tf.Variable]]):
    @classmethod
    def get_index(cls, container: List[tf.Variable], key: str) -> int:
        index = next(
            iter(
                [
                    i
                    for (i, c) in enumerate(container)
                    if f"{cls.get_item_name(c, i)}" == key
                ]
            ),
            None,
        )
        if index is None:
            raise KeyError(f"Could not find {key} on container")
        return index

    @classmethod
    def get_item_name(cls, container_item: tf.Variable, index: int) -> str:
        segments: List[str] = container_item.name.split("/", 2)
        name_component = segments[0].split("_")[0]
        if len(segments) > 1:
            return f"{name_component}_{index}/{segments[1]}"
        else:
            return f"{name_component}_{index}"


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
