__all__ = ["ChildArchitectureDetector"]

from typing import Dict, List

from jackdaw_ml.access_interface import (DefaultAccessInterface,
                                         DictAccessInterface)
from jackdaw_ml.access_interface.list_interface import ListAccessInterface
from jackdaw_ml.child_architecture import ChildArchitecture
from jackdaw_ml.detectors import ChildDetector
from jackdaw_ml.detectors.hook import DefaultDetectors

ChildArchitectureDetector = ChildDetector(
    child_models={
        ChildArchitecture: DefaultAccessInterface,
        List[ChildArchitecture]: ListAccessInterface,
        Dict[str, ChildArchitecture]: DictAccessInterface,
    }
)

DefaultDetectors.add_detector(ChildArchitectureDetector)
