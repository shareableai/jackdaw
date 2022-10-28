from jackdaw.child_architecture import ChildArchitecture
from jackdaw.detectors import Detector

TorchDetector = Detector(
    child_modules={ChildArchitecture},
    artefact_types=set(),
    serializer=None,
)
