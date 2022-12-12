# Jackdaw ML - Simplify Sharing Model Ops

We found that most models in production aren't from a single framework - SKLearn, PyTorch, Paddle, etc. - they're from a combination. 

Preprocessing in SKLearn, then clustering with PyTorch, then prediction with LightGBM, every additional model adds complexity to organisation and slows down deployment.

Which is why we developed Jackdaw - a model framework that discovers the parts of a model that need saving, and leaves the rest to you.

Documentation is baked into the repository, and is available [here](docs). 

## Setup - Working Locally
Jackdaw is available on [PyPi](https://pypi.org/project/jackdaw-ml/) and can be installed via pip;

```bash
>>> pip install jackdaw_ml
```

### Alpha - Limited Windows & Mac Support
While Jackdaw is in Alpha, one of the libraries it relies upon - artefactlink - only supports Windows and Mac OS/X for Python 3.10. Linux support is available for 3.8, 3.9, and 3.10.

## Setup - Sharing Models across Environments
To share items across multiple computers, you'll eventually need an account with ShareableAI. 

For now, you just need your API Token. If you don't have a token, reach out to `lissa@shareablei.com` and they'll ping you one.

## Roadmap - Future Features
[View our Public Roadmap here](https://github.com/orgs/shareableai/projects/1/views/1)


## Getting Started

### Example by Framework 
* [SKLearn](examples/frameworks/test_sklearn.py)
* [LightGBM](examples/frameworks/test_lightgbm.py)
* [XGBoost](examples/frameworks/test_xgboost.py)
* [PyTorch](examples/frameworks/test_pytorch.py)
* [Tensorflow](examples/frameworks/test_tensorflow.py)
* [DARTs](examples/frameworks/test_darts.py)

### Example

The core magic of Jackdaw is within the `@artefacts` and `@find_artefacts` decorators.

`@artefacts` allows you to list what should be saved on a Model. `@find_artefacts` will detect what should be saved based
on a whole host of common frameworks. Combining the two is a powerful way of ensuring complex models can be saved easily.


```python
import xgboost as xgb
import numpy as np

from jackdaw_ml.artefact_decorator import artefacts, find_artefacts
from jackdaw_ml.child_architecture import ChildArchitecture

from typing import Optional

@find_artefacts()
class MyXgbModel(ChildArchitecture):
    def __init__(self) -> None:
        self.xgb_model: Optional[xgb.Booster] = None
    
    def train(self, training_data: xgb.DMatrix) -> None:
        self.xgb_model = xgb.train({}, training_data)
        
    def predict(self, data: xgb.DMatrix) -> np.ndarray:
        return self.xgb_model.predict(data)

@artefacts()
class MyModel:
    def __init__(self):
        self.m1 = MyXgbModel()
        self.m2 = MyXgbModel()

    def evaluate(self, data: xgb.DMatrix):
        return (self.m1.predict(data) + self.m2.predict(data)) / 2.0
```

```python
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer
from jackdaw_ml.child_architecture import ChildArchitecture
from jackdaw_ml.trace import trace_artefacts

@artefacts({PickleSerializer: ["x"]})
class MySubModel(ChildArchitecture):
    def __init__(self):
        self.x = 3


@artefacts({})
class MyModel:
    def __init__(self):
        self.y = MySubModel()

# Create a new Model
model = MyModel()
# Modify the model
model.y.x = 4
# Save the model
model_id = model.dumps()

# # Current Artefacts on Model can be seen by calling `trace_artefacts`
# >>> trace_artefacts(model)
# <class '__main__.MyModel'>{
#        (y) <class '__main__.MySubModel'>{
#                (x) [<class 'jackdaw_ml.serializers.pickle.PickleSerializer'>]
#        }
#
# MyModel holds a child model on attribute 'y' called MySubModel, which contains a PickleSerialize'd artefact on 
#   attribute `x`

# Create another model
new_model = MyModel()
new_model.y.z = 10
# Load the model back in, using the Model ID
new_model.loads(model_id)
# New model is identical to the saved model
assert new_model.y.x == 4
# Non-artefacts aren't affected by `loads`
assert new_model.y.z == 10
```

## Detection

```python
from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer
from jackdaw_ml.artefact_decorator import find_artefacts
from jackdaw_ml.trace import trace_artefacts


@find_artefacts()
class MyIntModel:
    def __init__(self):
        self.x = 3


# As no automatic detectors exist for integers, `MyIntModel` shows no artefacts;
trace_artefacts(MyIntModel())


# >>> trace_artefacts(MyIntModel())
# <class '__main__.MyIntModel'>{}

@artefacts({PickleSerializer: ["x"]})
class MyIntModel:
    def __init__(self):
        self.x = 3


# Create a new Model
model = MyIntModel()
# Modify the model
model.y.x = 4
# Save the model
model_id = model.dumps()
```


## Saving Remotely
Saving and loading items from ShareableAI servers, rather than locally, can be achieved by providing an API key alongside the call to 
`artefacts`. If you'd like to test this, please follow the setup for Sharing Models Remotely.

```python

from jackdaw_ml.artefact_decorator import artefacts
from jackdaw_ml.serializers.pickle import PickleSerializer
from jackdaw_ml.artefact_endpoint import ArtefactEndpoint

@artefacts({PickleSerializer: ["x"]}, endpoint=ArtefactEndpoint.remote('MyAPIKey'))
class MyModel:
    def __init__(self):
        self.x = 3
```

We'll be expanding this to allow for sharing items between other users *very* soon, so keep an eye on [Corvus](https://github.com/shareableai/jackdaw/issues/2) to know more. 
