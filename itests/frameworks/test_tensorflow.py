import tensorflow as tf
import numpy as np

from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.detectors.keras import KerasSeqDetector, KerasDetector
from jackdaw_ml.debugging import artefact_debug

from functools import lru_cache
from typing import Tuple


mnist = tf.keras.datasets.mnist


@artefacts({})
class TFWrapper:
    model: tf.keras.models.Sequential

    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def fit(self, x_train, y_train, epochs) -> None:
        self.model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs)


@lru_cache(maxsize=1)
def example_train_data() -> Tuple:
    (x_train, y_train), (_) = mnist.load_data()
    x_train = x_train[0:100] / 255.0
    y_train = y_train[0:100]
    return x_train, y_train


@lru_cache(maxsize=1)
def example_test_data() -> Tuple:
    _, (x_test, _) = mnist.load_data()
    x_test = x_test / 255.0
    return x_test


def tf_float_equivalence(a: tf.Tensor, b: tf.Tensor) -> bool:
    return np.all((tf.abs(a - b) < 0.000001).numpy())


def model_equivalent(m1: TFWrapper, m2: TFWrapper) -> bool:
    m1_res = m1.model(example_test_data())
    m2_res = m2.model(example_test_data())
    return tf_float_equivalence(m1_res, m2_res)


def test_basic_wrapper():
    m1 = TFWrapper()
    m1.fit(*example_train_data(), epochs=1)
    model_id = m1.dumps()
    m2 = TFWrapper()
    m2.loads(model_id)
    assert model_equivalent(m1, m2)
