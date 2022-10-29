# Detectors
Detectors are ways of identifying artefacts and child models within a model. They avoid the need for manually specifying
each artefact on a class, but may need some tweaking for models that they haven't seen before.

Examples of using Detectors are available in [docs/save](./save.md).

New detectors can be created using the generic `Detector` initialiser, which requires up to 4 components; 

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass(slots=True)
class Detector(Generic[T]):
    """
    Generic Detector for Child Models and Artefacts that require a specific serializer

    Attributes
    ----------
    `child_models`
        Types or Classes to be detected as Child Models

    `artefact_types`
        Types or Classes to be detected as Artefacts

    `serializer`
        Class to Serialize an object from types in `artefact_types` to Bytes

    `storage_location`
        If set, identify the attribute on the module which `get` and `set` will use as a target.
        If not set, items will be set and retrieved from the `__dict__` attribute expected on
        all objects.
        If set and the object is of type `dict`, it will be replaced with an ArtefactDict
        to ensure it is possible to set methods on the class for `loads`/`dumps` etc.
    """
```

### Child Models
Child Models are models that are required by any other model that parents it. You can use this to build up models
while still having a single class item, and it can be a useful wrapper when working with different model frameworks, i.e. 


```python
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer
from jackdaw_ml.serializers.tensor import TorchSerializer
from jackdaw_ml.detectors.torch import TorchDetector

import torch.nn as nn

class MyPreProcessor:
    ...

@artefacts({PickleSerializer: ['x']})
class PreProcModel(MyPreProcessor):
    x: int = 5

@artefacts({TorchSerializer: ['fc1', 'fc2']})
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
@artefacts({}, detectors=[TorchDetector])
class MyModel:
    preprocessor: PreProcModel
    mlp: TorchModel
```

`PreProcModel` uses a different framework entirely to TorchModel, I want to reliably save both of them in `MyModel` without
specifying PreProcModel as a ChildArchitecture each time. This means that my Detector will need to know about the preprocessor class, i.e.

```python
from jackdaw_ml.detectors import Detector

MyDetector = Detector(child_models={MyPreProcessor}, ...)
```

Any class that is a class or subclass of `MyPreProcessor` will now be detected as a ChildModel. 

More complex detection examples can be found in [Torch Geometric Detectors](../jackdaw_ml/detectors/torch_geo.py), where
the detector is fully custom, as checking for class equality isn't sufficient.

### Artefact Types
What if we also had some custom items that couldn't be saved with a PickleSerializer? Let's teach a Detector how to identify
them, and also how to save and load them. 

For writing your own Serializer, check out [docs/serializers.md](serializers.md)

```python

class MyItem:
    x:  = 5
```

With our new `MyItemSerializer`, we can indicate to Jackdaw we'd like to use it over all instances of `MyItem` by adding to 
our detector;

```python
from jackdaw_ml.detectors import Detector
from jackdaw_ml.serializers import Serializable

class MyPreProcessor: ...
class MyItem: ...
class MyItemSerializer(Serializable[MyItem]): ...


MyDetector = Detector(child_models={MyPreProcessor}, artefact_types={MyItem}, serializer=MyItemSerializer)
```

### Storage Location
Most classes in Python store their attributes in `__dict__`, and so taking and placing objects within the `__dict__` is sufficient
for Jackdaw. Some, like TorchSeq, don't. To get around this, Detectors are allowed to look in different places on the object
to find objects that class holds, such as `_modules`, or any other location.


You can then use your detector like any other detector in calls to @artefacts.