__all__ = ["ChildArchitectureDetector"]

from jackdaw_ml.child_architecture import ChildArchitecture
from jackdaw_ml.detectors import Detector
from jackdaw_ml.detectors.hook import DefaultDetectors

ChildArchitectureDetector = Detector(
    child_models={ChildArchitecture},
    artefact_types=set(),
    serializer=None,
)

DefaultDetectors.add_detector(ChildArchitectureDetector)
