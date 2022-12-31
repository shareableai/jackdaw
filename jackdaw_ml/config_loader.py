import os
from typing import List


def _add_init(cls) -> None:
    items = {
        item_name: eval(item_class)(os.getenv(item_name))
        for item_name, item_class in cls.__dict__.get("__annotations__", {})._items()
    }

    def __init__(self) -> None:
        for (item_name, item) in items.items():
            setattr(self, item_name, item)

    setattr(cls, "__init__", __init__)


def from_env(cls=None, *args: List[str], **kwargs):
    def wrap(cls):
        _add_init(cls)
        return cls

    if cls is None:
        return wrap
    return wrap(cls)
