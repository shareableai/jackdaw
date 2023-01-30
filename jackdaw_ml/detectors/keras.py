__all__ = ["KerasSeqDetector", "KerasArtefactDetector"]

from typing import List

import keras
import tensorflow as tf

from jackdaw_ml.access_interface import AccessInterface, DefaultAccessInterface
from jackdaw_ml.access_interface.list_interface import ListAccessInterface
from jackdaw_ml.detectors import ArtefactDetector, ChildDetector
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.keras import KerasSerializer


class KerasLayerAccessInterface(
    ListAccessInterface[AccessInterface[List[tf.Variable], tf.Variable]]
):
    @classmethod
    def get_index(cls, container: List[tf.Variable], key: str) -> int:
        index = next(
            (
                i
                for (i, c) in enumerate(container)
                if f"{cls.get_item_name(c, i)}" == key
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


KerasSeqDetector = ChildDetector(
    child_models={
        List[keras.layers.Layer]: KerasLayerAccessInterface,
        keras.layers.Layer: DefaultAccessInterface,
    }
)

KerasArtefactDetector = ArtefactDetector(
    artefact_types={tf.Variable},
    serializer=KerasSerializer,
)


DefaultDetectors.add_detector(KerasSeqDetector, DetectionLevel.Specific)
DefaultDetectors.add_detector(KerasArtefactDetector, DetectionLevel.Generic)
