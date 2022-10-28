# ShareableAI - Simplify Sharing Models

## Setup - Working Locally
The core library is available on [PyPi](https://pypi.org/project/jackdaw/) and can be installed via pip;

```bash
>>> pip install jackdaw_ml
```
## Setup - Sharing Models Remotely
To share items remotely, you'll eventually need an account with ShareableAI. 

For now, you just need your API Token. If you don't have a token, reach out to `lissa@shareablei.com` and they'll ping you one.  Add your token to a local config file (~/.shareableai/credentials), and you're good to go.


## Working with Local Models
### Saving Generic Models

With simple models, saving the model can be as simple as specifying which items in a class should be saved.

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
```python
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

### Saving PyTorch Models



### Saving SKLearn Models


## Sharing Models