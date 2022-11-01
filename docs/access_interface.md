# Access Interfaces / Jackdaw Design Decisions

Python objects, by and large, are all classes and contain attributes. Jackdaw intends on finding 
these attributes relevant to ML models, serializing them to disk, and marking down where they came from.

When it comes to putting things back, Jackdaw then has an easy time putting things back where they were
marked as coming from. In the simplest case this is similar to a dataclasses style class, i.e.


```python
class X:
    y: int = 5
    z: int = 10
```

Where `y` and `z` are easy to demarcate as items belonging to X. If Jackdaw wants to access them directly,
it can use `getattr` to pull the attribute. Jackdaw can also assume they exist on the `__dict__` attribute
of the class where attributes of the instance of the class exist, and access them like a dictionary. 

Sometimes attributes exist on a class, but aren't on the `__dict__` attribute, but on another location, 
like `_modules` on Torch's `nn.ModuleList` and `nn.Sequential`. This isn't particularly complex, and can
be addressed by changing the location to run `getattr` from `__dict__` to `_modules`.

Where it becomes especially complex is for containers that aren't implementing a dict-like structure, 
such as `List`. In those cases Jackdaw must either treat each container specially, or implement an 
abstract interface over each container that allows the generic behaviour to continue. I am assuming that
this generic behaviour remains possible, and so implement a `AccessInterface` over containers that allows
for this type of "everything has a named box" type of data storage and retrieval.