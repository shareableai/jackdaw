__all__ = ["DefaultDetectors", "DetectionLevel"]

from jackdaw_ml.detectors import Detector

from typing import Dict
from enum import Enum, auto


class DetectionLevel(Enum):
    Specific = auto()
    Generic = auto()


class DefaultDetectors:
    _specific_detectors: Dict[Detector, None] = dict()
    _generic_detectors: Dict[Detector, None] = dict()

    @staticmethod
    def add_detector(
        detector: Detector, level: DetectionLevel = DetectionLevel.Generic
    ):
        match level:
            case DetectionLevel.Generic:
                DefaultDetectors._generic_detectors[detector] = None
            case DetectionLevel.Specific:
                DefaultDetectors._specific_detectors[detector] = None

    @staticmethod
    def detectors() -> Dict[Detector, None]:
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
        return (
            DefaultDetectors._specific_detectors | DefaultDetectors._generic_detectors
        )
