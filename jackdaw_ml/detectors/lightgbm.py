__all__ = ["LightGBMDetector"]

import logging

LOGGER = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    LOGGER.error(
        "Could not load LightGBM required for LightGBMDetector - please ensure LightGBM is installed."
    )

from jackdaw_ml.detectors import ArtefactDetector
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.pickle import PickleSerializer

LightGBMDetector = ArtefactDetector(
    artefact_types={lgb.Booster}, serializer=PickleSerializer
)


DefaultDetectors.add_detector(LightGBMDetector, DetectionLevel.Specific)
