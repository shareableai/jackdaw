# Saving Models

## What is actually Saved?
Jackdaw intends to save *minimally*, only saving the components that are required for the model to function. It achieves
this through the use of `Detectors` - ways of identifying if an item is a Model, a Child Model, or an `Artefact`. Models and Child Models 
are classes that contain `Artefacts`, and `Artefacts` are the items that are saved and loaded.

You can also skip detectors if you'd like, and manually mark which items should be saved on each class. This can help if 
Jackdaw is struggling to identify a specific object, or there isn't a suitable `detector`. (Although we'd like to note that user defined 
detectors are also an option!)

Minimalistic saving improves the robustness of restoring models - if we pickled an entire class, then we're also responsible
for all the methods on the class, all the requirements for that class, etc. If we only save the numbers in a parameter, we can
guarantee the model is loadable in almost any situation, even if you change Python versions, change OS, or change the rest of the model
code.

The other advantage is that we only save objects we haven't seen before. If you're expanding a pre-existing multi-gigabyte 
model by adding a new layer, we will only save the new layer. You should see huge improvements in your model save times, 
without changing any of your behaviour.

## Saving Local Models
With simple models, saving the model can be as simple as specifying which items in a class should be saved. For this model 
we use a [PickleSerializer](../jackdaw_ml/serializers/pickle.py) to save the item `initially_true` on the `ModelExample`.

```python
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer

@artefacts({PickleSerializer: "initially_true"})
class ModelExample:
    def __init__(self) -> None:
        self.initially_true = True
        self.initially_false = False

model = ModelExample()
model.initially_true = False

# Save the model locally
model_identifier = model.dumps()
```

```bash
# Interactively, create a new model
>>> new_model = ModelExample()
# Check that the initial state is True
>>> print(new_model.initially_true)
True
# Load the model back
>>> new_model.loads(model_identifier)
# State is returned to our saved case
>>> print(new_model.initially_true)
False
```

## Specialisations - PyTorch
Some models have specialised save/load options to make saving simpler. PyTorch is one of these, and has a `TorchDetector`. 
Sequences in PyTorch are a little different, such as nn.Sequential and nn.ModuleDict, and have a `TorchSeqDetector`. If you're 
unsure which should be used, more detectors are usually better than too few. 

```python
from jackdaw_ml.artefact_decorator import  artefacts
from jackdaw_ml.detectors.torch import TorchDetector, TorchSeqDetector

@artefacts(detectors=[TorchDetector, TorchSeqDetector, ...])
```

## Nested Models
Detection becomes incredibly useful for nested models, where specifying artefacts at multiple levels becomes overly verbose. 

```python
import torch.nn as nn
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.detectors.torch import TorchDetector

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

@artefacts(detectors=[TorchDetector])
class MyNestedNet(nn.Module):
    def __init__(self):
        super(MyNestedNet, self).__init__()
        self.net1 = Net()
        self.net2 = Net()
```

Jackdaw will identify `Net` as a Child Model, and `MyNestedNet` as the Parent. Within the Child Model, Jackdaw will find 
`fc1` and `fc2` using the Torch Detector, and save the inner parts of those components. 

When it comes to generic models, you can also indicate that a model is a child model by subclassing it as a ChildArchitecture, such as; 

```python
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer
from jackdaw_ml.child_architecture import ChildArchitecture

@artefacts({PickleSerializer: ["x"]})
class MySubModel(ChildArchitecture):
    def __init__(self):
        self.x = 3


@artefacts({})
class MyModel:
    def __init__(self):
        self.y = MySubModel()
```

Now Jackdaw will detect `MySubModel` as an item that contains artefacts.