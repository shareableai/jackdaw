# Jackdaw ML - Simplify Sharing Models

Jackdaw ML is a tool of ShareableAI to make it easier to share your Machine Learning models. Jackdaw is currently in pre-alpha, and 
shouldn't be used in production by anyone.

Documentation is baked into the repository, and is available [here](docs). 

## Setup - Working Locally
The core library is available on [PyPi](https://pypi.org/project/jackdaw-ml/) and can be installed via pip;

```bash
>>> pip install jackdaw_ml
```
## Setup - Sharing Models Remotely
To share items remotely, you'll eventually need an account with ShareableAI. 

For now, you just need your API Token. If you don't have a token, reach out to `lissa@shareablei.com` and they'll ping you one.  Add your token to a local config file (~/.shareableai/credentials), and you're good to go.

## Roadmap - Future Features
[View our Public Roadmap here](https://github.com/orgs/shareableai/projects/1/views/1)


## Getting Started Example

Below we save a simple model with Jackdaw. Examples for PyTorch can be found in [our tests](itests/test_e2e.py), and more 
information can be found in [our documentation](docs/save.md)

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

# Create a new Model
model = MyModel()
# Modify the model
model.y.x = 4
# Save the model
model_id = model.dumps()

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