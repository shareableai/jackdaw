from jackdaw_ml.artefact_decorator import artefacts


@artefacts()
class MyModel:
    def __init__(self):
        self.x = 5


def test_save_metric():
    model = MyModel()
    model._log_metric("MyMetric", 0)
