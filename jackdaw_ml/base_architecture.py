# TODO[Optional]: Move Architecture(s) to another Python module.

__all__ = ["BaseArchitecture"]

from typing import AsyncIterator

# Design with bidirectional approaches in-mind
DataGenerator = AsyncIterator


class BaseArchitecture:
    pass
