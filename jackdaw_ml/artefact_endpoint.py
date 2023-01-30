from __future__ import annotations

__all__ = ["ArtefactEndpoint"]

from dataclasses import dataclass
from typing import Union

from artefact_link import (LocalArtefactRegistry, LocalEndpoint,
                           ShareableAIEndpoint)


@dataclass
class ArtefactEndpoint:
    endpoint: Union[LocalEndpoint, ShareableAIEndpoint]
    """ Artefact Endpoint
    Specifies where to store Artefact data - either on a local path or on ShareableAI resources.
    
    SAI Resources requires an API Key, but storing locally will just use default file locations
    from Home. 
    """

    @staticmethod
    def remote(api_key: str):
        return ArtefactEndpoint(ShareableAIEndpoint(api_key))

    @staticmethod
    def default() -> ArtefactEndpoint:
        """
        Create an Artefact Endpoint using default values - this will use
        a local SQLite server and create local storage on the filesystem, with
        the SQLite server at ~/.artefact_registry and the storage at
        ~/.artefact_storage
        """
        return ArtefactEndpoint(
            LocalEndpoint(
                registry_endpoint=LocalArtefactRegistry(None), storage_location=None
            )
        )
