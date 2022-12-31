from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.loads import loads
from jackdaw_ml.saves import saves
from jackdaw_ml.serializers.pickle import PickleSerializer
from jackdaw_ml.child_architecture import ChildArchitecture


@artefacts({PickleSerializer: ["x"]})
class MySubModel(ChildArchitecture):
    def __init__(self):
        self.x = 3


@artefacts({})
class MyModel:
    def __init__(self):
        self.y = MySubModel()


def test_nested_model():
    model = MyModel()
    model.y.x = 4
    model_id = saves(model)

    new_model = MyModel()
    loads(new_model, model_id)
    assert new_model.y.x == 4
