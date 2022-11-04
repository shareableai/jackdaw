from functools import lru_cache

import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from jackdaw_ml.artefact_decorator import artefacts
import tensorflow as tf

mnist = tf.keras.datasets.mnist


@artefacts({})
class MixedModel:
    sklearn_preproc: StandardScaler
    tf_model: tf.keras.models.Sequential

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def fit(self, x_train, y_train, epochs) -> None:
        self.model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        x_train = self.scaler.fit_transform(np.reshape(x_train, (x_train.shape[0], 28*28)))
        x_train = np.reshape(x_train, (x_train.shape[0], 28, 28))
        self.model.fit(x_train, y_train, epochs)

    def predict(self, x_test):
        x_test = self.scaler.fit_transform(np.reshape(x_test, (x_test.shape[0], 28 * 28)))
        x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
        return self.model(x_test)


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


def model_equivalent(m1: MixedModel, m2: MixedModel) -> bool:
    m1_res = m1.predict(example_test_data())
    m2_res = m2.predict(example_test_data())
    return tf_float_equivalence(m1_res, m2_res)


def test_basic_wrapper():
    m1 = MixedModel()
    m1.fit(*example_train_data(), epochs=1)
    model_id = m1.dumps()
    m2 = MixedModel()
    m2.loads(model_id)
    assert model_equivalent(m1, m2)
