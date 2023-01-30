from __future__ import annotations

__all__ = ["Resource"]

import pathlib
import tempfile
from hashlib import md5
from io import BytesIO
from typing import Optional, SupportsBytes, Union

from artefact_link import PyArtefact


class Resource:
    inner: bytes
    inner_hash: Optional[int]

    def __init__(self, bytes_like: Union[bytes, SupportsBytes, BytesIO]):
        if isinstance(bytes_like, SupportsBytes):
            self.inner = bytes_like.__bytes__()
        elif isinstance(bytes_like, BytesIO):
            bytes_like.seek(0)
            self.inner = bytes_like.getvalue()
        else:
            self.inner = bytes_like
        self.inner_hash = None

    def __bytes__(self) -> bytes:
        return self.inner

    def __hash__(self) -> int:
        if self.inner_hash is None:
            self.inner_hash = int(md5(self.inner).hexdigest(), base=16)
        return self.inner_hash

    @staticmethod
    def from_artefact(artefact: PyArtefact) -> Resource:
        with tempfile.TemporaryDirectory() as t:
            return Resource(open(artefact.path(pathlib.Path(t)), "rb").read())
