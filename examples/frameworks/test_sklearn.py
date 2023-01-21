from functools import lru_cache

import numpy as np
from typing import Tuple

from sklearn.ensemble import IsolationForest
from jackdaw_ml.artefact_decorator import find_artefacts
from jackdaw_ml import loads
from jackdaw_ml import saves


@find_artefacts()
class BasicSklearnWrapper:
    """
    Replicating SKLearn's interface here for convenience
    """

    model: IsolationForest

    def __init__(self):
        self.model = IsolationForest()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)


def example_data() -> Tuple[np.ndarray, np.ndarray]:
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    return data, label


@lru_cache(maxsize=1)
def example_prediction_data() -> np.ndarray:
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    return data


def np_float_equivalence(a: np.ndarray, b: np.ndarray) -> bool:
    # Are `a` and `b` within a reasonable distance of each other, accounting for internal machine error?
    return np.sum(a - b) <= np.finfo(np.float32).eps


def test_sklearn_equivalence_with_prefit():
    m1 = BasicSklearnWrapper()
    m2 = BasicSklearnWrapper()
    m1.fit(*example_data())
    m1.predict(example_prediction_data())
    model_id = saves(m1)
    m2.fit(*example_data())
    loads(m2, model_id)

    assert np_float_equivalence(
        m1.predict(example_prediction_data()), m2.predict(example_prediction_data())
    )

# Failing due to inability of Jackdaw to instantiate new classes when required.
# Looking at tooling like https://github.com/eevee/camel/blob/master/camel/__init__.py to resolve this using
# a more generic version of Jackdaw's serialization system.
def test_sklearn_equivalence():
    m1 = BasicSklearnWrapper()
    m2 = BasicSklearnWrapper()
    m1.fit(*example_data())
    m1.predict(example_prediction_data())
    model_id = saves(m1)
    loads(m2, model_id)

    assert np_float_equivalence(
        m1.predict(example_prediction_data()), m2.predict(example_prediction_data())
    )
