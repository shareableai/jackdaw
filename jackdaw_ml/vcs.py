from __future__ import annotations

__all__ = ["get_vcs_info"]

import logging
import pathlib

from git import Repo, InvalidGitRepositoryError
from gitdb.exc import BadName
from giturlparse import parse as git_parse

from artefact_link import PyVcsInfo, PyRemoteRepository

LOGGER = logging.getLogger(__name__)

_PARENTS_SEARCH = 10


def get_vcs_info(path=pathlib.Path.cwd(), parent_search: int = 0) -> PyVcsInfo:
    try:
        repo = Repo(path)
    except InvalidGitRepositoryError:
        if parent_search > _PARENTS_SEARCH:
            LOGGER.warning("Could not find Git Repo")
            return PyVcsInfo("NoSHAProvided", None)
        else:
            return get_vcs_info(path.parent, parent_search + 1)
    if repo.is_dirty():
        LOGGER.warning(
            "Current Repo is dirty - this hash may not represent the current working state"
        )
    remote = next(map(lambda x: git_parse(str(x.url)), repo.remotes), None)
    try:
        return PyVcsInfo(
            str(repo.commit("HEAD")),
            PyRemoteRepository(remote.resource, remote.name, remote.owner)
            if remote
            else None,
        )
    except BadName:
        return PyVcsInfo(
            "DirtyHashState",
            PyRemoteRepository(remote.resource, remote.name, remote.owner)
            if remote
            else None,
        )
