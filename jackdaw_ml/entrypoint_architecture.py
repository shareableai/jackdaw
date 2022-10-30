__all__ = ["EntrypointArchitecture"]

from jackdaw_ml.base_architecture import BaseArchitecture, DataGenerator


class EntrypointArchitecture(BaseArchitecture):
    """
    External Entrypoint to a Model

    Allows for an API to be constructed against this Model, so that external
    data may be passed to it.
    """

    def train(self, data_generator: DataGenerator):
        raise NotImplementedError

    def predict(self, data_generator: DataGenerator) -> DataGenerator:
        raise NotImplementedError
