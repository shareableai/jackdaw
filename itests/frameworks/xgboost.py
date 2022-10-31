import json
import tempfile

import numpy as np
import xgboost as xgb

from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer

from functools import lru_cache


@artefacts({PickleSerializer: ['model']})
class BasicXGBWrapper:
    """
    LightGBM is zipsafe, so there's no real issue with using PickleSerializer over the Booster objects it provided.
    For better performance, you can construct a custom LGB serializer.
    """
    booster: xgb.Booster


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


def model_equivalent(m1: BasicXGBWrapper, m2: BasicXGBWrapper) -> bool:
    with tempfile.NamedTemporaryFile('wb') as m1_f:
        with tempfile.NamedTemporaryFile('wb') as m2_f:
            m1_res = m1.model.predict(example_data_raw())
            m2_res = m2.model.predict(example_data_raw())
            m1_f.close()
            m2_f.close()
            m1.model.dump_model(str(m1_f.name))
            m2.model.dump_model(str(m2_f.name))
    return np_float_equivalence(m1_res, m2_res) and open(m1_f.name).read() == open(m2_f.name).read()


def test_basic_wrapper():
    m1 = BasicXGBWrapper()
    m1.model = xgb.train({}, example_data())
    model_id = m1.dumps()
    m2 = BasicXGBWrapper()
    m2.loads(model_id)
    assert model_equivalent(m1, m2)

