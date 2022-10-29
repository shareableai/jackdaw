# Serializing Objects

Serializing is just being able to save an object to bytes, and load it back. 

If we can do
```
class MyItem: ...

MyItem -> bytes
bytes -> MyItem
```

Then we have a basic Serializer. This is incredibly useful for how Jackdaw wants to tackle saving models - it wants to 
save minimally and needs to know how to work with each Artefact.


If we look at the [PickleSerializer](../jackdaw_ml/serializers/pickle.py) it should be fairly clear how to create new ones. 

```python
import pickle
from typing import TypeVar, Optional

from jackdaw_ml.resource import Resource
from jackdaw_ml.serializers import Serializable

T = TypeVar("T")

class PickleSerializer(Serializable[T]):
    @staticmethod
    def to_resource(item: T) -> Resource:
        return Resource(pickle.dumps(item))

    @staticmethod
    def from_resource(uninitialised_item: Optional[T], buffer: Resource) -> T:
        return pickle.loads(buffer.__bytes__())
```

Ignoring `Serializable` for now, `PickleSerializer` has two functions that we described earlier. Move an item to bytes (`Resource` is basically a fancy bytes container), 
and then move it back. Pickle takes care of most of this, so saving to bytes is dumping the object using `pickle.dumps`, and loading from bytes is `pickle.loads`. 

The only additional component is `uninitialised_item`. When Jackdaw comes to a model, it might already find a class there to work on, and be able to modify it, rather 
than replacing it entirely. Pickle saves the entire item wholesale, so our PickleSerializer replaces the item, but it doesn't mean you have to. 