import tensorflow as tf

from jackdaw_ml.access_interface import DefaultAccessInterface
from jackdaw_ml.artefact_decorator import _add_artefacts
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint
from jackdaw_ml.detectors.hook import DefaultDetectors
from jackdaw_ml.detectors.keras import KerasLayerAccessInterface


def test_layer_detector():
    x = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    _add_artefacts(x, ArtefactEndpoint.default(), {})
    res = dict(
        DefaultAccessInterface.list_children(
            x,
            list(DefaultDetectors.child_detectors().keys()),
            list(DefaultDetectors.artefact_detectors().keys()),
        )
    )
    assert len(dict(res).keys()) > 0


def test_layer_recursion():
    x = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    _add_artefacts(x, ArtefactEndpoint.default(), {})
    res = dict(
        KerasLayerAccessInterface.list_children(
            x.layers,
            list(DefaultDetectors.child_detectors().keys()),
            list(DefaultDetectors.artefact_detectors().keys()),
        )
    )
    assert len(dict(res).keys()) == 4
