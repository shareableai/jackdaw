__all__ = ["SKLearnDetector"]

import logging

from jackdaw_ml.access_interface.sklearn_interface import SkLearnInterface

LOGGER = logging.getLogger(__name__)

try:
    import sklearn
except ImportError:
    LOGGER.error(
        "Could not load Scikit-Learn required for SKLearnDetector - please ensure scikit-learn is installed."
    )

from jackdaw_ml.detectors import ChildDetector
from jackdaw_ml.detectors.hook import DefaultDetectors, DetectionLevel


"""
    Typically frameworks don't get their own interfaces. 
    SKLearn gets one because it reuses base Python so heavily it's hard to tell it apart from other items.
"""

SKLearnDetector = ChildDetector(
    child_models={sklearn.base.BaseEstimator: SkLearnInterface}
)


DefaultDetectors.add_detector(SKLearnDetector, DetectionLevel.Generic)
