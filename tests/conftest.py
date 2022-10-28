import logging
import os
from typing import *

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

T = TypeVar("T")

serializable_items = [
    0,
    list(range(10_000)),
    "a",
    ["a", "b"],
    set(list(range(10_000))),
    {"a": [1, 2, 3]},
]


def take_n(a: Iterable[T], items: int) -> Iterator[T]:
    a = iter(a)
    for _ in range(items):
        yield next(a)


TEST_API_KEY = os.getenv("SHAREABLEAI_TEST_API_KEY", "Empty")
