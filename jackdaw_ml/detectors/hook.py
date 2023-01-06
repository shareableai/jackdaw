__all__ = ["DefaultDetectors", "DetectionLevel"]

from jackdaw_ml.detectors import Detector, ArtefactDetector, ChildDetector

from typing import Dict, Union
from enum import Enum, auto


class DetectionLevel(Enum):
    Specific = auto()
    Generic = auto()


class DefaultDetectors:
    _specific_detectors: Dict[Union[ArtefactDetector, ChildDetector], None] = dict()
    _generic_detectors: Dict[Union[ArtefactDetector, ChildDetector], None] = dict()

    @staticmethod
    def add_detector(
        detector: Union[ArtefactDetector, ChildDetector],
        level: DetectionLevel = DetectionLevel.Generic,
    ):
        match level:
            case DetectionLevel.Generic:
                DefaultDetectors._generic_detectors[detector] = None
            case DetectionLevel.Specific:
                DefaultDetectors._specific_detectors[detector] = None

    @staticmethod
    def artefact_detectors() -> Dict[ArtefactDetector, None]:
        try:
            from jackdaw_ml.detectors.torch import TorchDetector, TorchSeqDetector
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.keras import KerasDetector, KerasSeqDetector
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.torch_geo import TorchGeoSeqDetector
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.child_architecture_detector import (
                ChildArchitectureDetector,
            )
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.lightgbm import (
                LightGBMDetector,
            )
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.sklearn import (
                SKLearnDetector,
            )
        except ImportError:
            pass
        return {
            detector: None
            for detector in (
                DefaultDetectors._specific_detectors
                | DefaultDetectors._generic_detectors
            )
            if isinstance(detector, ArtefactDetector)
        }

    @staticmethod
    def child_detectors() -> Dict[ChildDetector, None]:
        try:
            from jackdaw_ml.detectors.torch import TorchDetector, TorchSeqDetector
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.keras import KerasDetector, KerasSeqDetector
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.torch_geo import TorchGeoSeqDetector
        except ImportError:
            pass
        try:
            from jackdaw_ml.detectors.child_architecture_detector import (
                ChildArchitectureDetector,
            )
        except ImportError:
            pass
        return {
            detector: None
            for detector in (
                DefaultDetectors._specific_detectors
                | DefaultDetectors._generic_detectors
            )
            if isinstance(detector, ChildDetector)
        }
