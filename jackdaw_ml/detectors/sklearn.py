__all__ = ["SKLearnDetector"]

import logging

LOGGER = logging.getLogger(__name__)

try:
    import sklearn
except ImportError:
    LOGGER.error(
        "Could not load Scikit-Learn required for SKLearnDetector - please ensure scikit-learn is installed."
    )

from jackdaw_ml.detectors import ArtefactDetector
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel
from jackdaw_ml.serializers.pickle import PickleSerializer

SKLearnDetector = ArtefactDetector(
    artefact_types={sklearn.base.BaseEstimator}, serializer=PickleSerializer
)


DefaultDetectors.add_detector(SKLearnDetector, DetectionLevel.Generic)
