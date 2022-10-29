from jackdaw_ml.artefact_decorator import artefacts
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
    model_id = model.dumps()

    new_model = MyModel()
    new_model.loads(model_id)
    assert new_model.y.x == 4
