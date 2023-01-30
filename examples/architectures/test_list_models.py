import tempfile
from functools import lru_cache
from typing import List, Optional

import numpy as np
import xgboost as xgb

from jackdaw_ml import loads, saves
from jackdaw_ml.artefact_decorator import artefacts, find_artefacts
from jackdaw_ml.child_architecture import ChildArchitecture
from jackdaw_ml.serializers.pickle import PickleSerializer


@artefacts({PickleSerializer: ["xgb_model"]})
class MyXgbModel(ChildArchitecture):
    def __init__(self) -> None:
        self.xgb_model: Optional[xgb.Booster] = None

    def train(self, training_data: xgb.DMatrix) -> None:
        self.xgb_model = xgb.train({}, training_data)

    def predict(self, data: xgb.DMatrix) -> np.ndarray:
        return self.xgb_model.predict(data)


@find_artefacts()
class MyModel:
    def __init__(self, n_models: int = 5):
        self.models: List[MyXgbModel] = [MyXgbModel() for _ in range(n_models)]

    def train(self, data: xgb.DMatrix) -> None:
        for model in self.models:
            model.train(data)

    def predict(self, data: xgb.DMatrix) -> np.ndarray:
        return np.mean(np.array([m.predict(data) for m in self.models]), 0)


@lru_cache(maxsize=1)
def example_data() -> xgb.DMatrix:
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    return xgb.DMatrix(data, label=label)


@lru_cache(maxsize=1)
def example_data_raw() -> xgb.DMatrix:
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    return xgb.DMatrix(data)


def np_float_equivalence(a: np.ndarray, b: np.ndarray) -> bool:
    # Are `a` and `b` within a reasonable distance of each other, accounting for internal machine error?
    return np.sum(a - b) <= np.finfo(np.float32).eps


def model_equivalent(m1: MyModel, m2: MyModel) -> bool:
    with tempfile.NamedTemporaryFile("wb") as m1_f:
        with tempfile.NamedTemporaryFile("wb") as m2_f:
            m1_res = m1.predict(example_data_raw())
            m2_res = m2.predict(example_data_raw())
            m1_f.close()
            m2_f.close()
            m1.models[0].xgb_model.dump_model(str(m1_f.name))
            m2.models[0].xgb_model.dump_model(str(m2_f.name))
    return (
        np_float_equivalence(m1_res, m2_res)
        and open(m1_f.name).read() == open(m2_f.name).read()
    )


def test_list_wrapper():
    m1 = MyModel()
    m1.train(example_data())
    model_id = saves(m1)
    m2 = MyModel()
    loads(m2, model_id)
    assert model_equivalent(m1, m2)
