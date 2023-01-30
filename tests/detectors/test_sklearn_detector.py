from functools import lru_cache
from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from jackdaw_ml import loads, saves
from jackdaw_ml.artefact_decorator import find_artefacts


@find_artefacts()
class BasicSklearnWrapper:
    model: IsolationForest
    preproc: StandardScaler

    def __init__(self):
        self.preproc = StandardScaler()
        self.model = IsolationForest()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self.preproc.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.preproc.transform(X)
        return self.model.predict(X)


@lru_cache(maxsize=1)
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
