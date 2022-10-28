from jackdaw_ml.child_architecture import ChildArchitecture
from jackdaw_ml.detectors import Detector

TorchDetector = Detector(
    child_modules={ChildArchitecture},
    artefact_types=set(),
    serializer=None,
)
