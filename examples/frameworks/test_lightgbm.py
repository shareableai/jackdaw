import json

import numpy as np
import lightgbm as lgb

from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.loads import loads
from jackdaw_ml.saves import saves
from jackdaw_ml.serializers.pickle import PickleSerializer

from functools import lru_cache


@artefacts(artefact_serializers={PickleSerializer: ["model"]})
class BasicLGBWrapper:
    model: lgb.Booster


@lru_cache(maxsize=1)
def example_data() -> lgb.Dataset:
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    return lgb.Dataset(data, label=label)


@lru_cache(maxsize=1)
def example_data_raw() -> np.ndarray:
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    return data


def np_float_equivalence(a: np.ndarray, b: np.ndarray) -> bool:
    # Are `a` and `b` within a reasonable distance of each other, accounting for internal machine error?
    return np.sum(a - b) <= np.finfo(np.float32).eps


def model_equivalent(m1: BasicLGBWrapper, m2: BasicLGBWrapper) -> bool:
    m1_res = m1.model.predict(example_data_raw())
    m2_res = m2.model.predict(example_data_raw())
    m1_dump = m1.model.dump_model()
    m2_dump = m2.model.dump_model()
    return np_float_equivalence(m1_res, m2_res) and json.dumps(m1_dump) == json.dumps(
        m2_dump
    )


def test_basic_wrapper():
    m1 = BasicLGBWrapper()
    m1.model = lgb.train({}, example_data())
    model_id = saves(m1)
    m2 = BasicLGBWrapper()
    loads(m2, model_id)
    assert model_equivalent(m1, m2)
