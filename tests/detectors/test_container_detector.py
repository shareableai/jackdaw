import random
from typing import List

from jackdaw_ml import loads, saves
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.detectors import ArtefactDetector
from jackdaw_ml.serializers.pickle import PickleSerializer

IntDetector = ArtefactDetector(
    artefact_types={int},
    serializer=PickleSerializer,
)


@artefacts(artefact_detectors=[IntDetector])
class MyModel:
    def __init__(self, n_models: int = 5):
        self.models: List[int] = [random.randint(0, 10) for _ in range(n_models)]

    def predict(self, data: int) -> int:
        return sum([m * data for m in self.models])


def test_detection():
    m1 = MyModel()
    model_id = saves(m1)
    m2 = MyModel()
    loads(m2, model_id)
    assert m1.predict(2) == m2.predict(2)
    assert m1.models == m2.models
