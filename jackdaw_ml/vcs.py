from __future__ import annotations

__all__ = ["get_vcs_info"]

import logging
import pathlib

from artefact_link import PyRemoteRepository, PyVcsInfo
from git import InvalidGitRepositoryError, Repo
from gitdb.exc import BadName
from giturlparse import parse as git_parse

LOGGER = logging.getLogger(__name__)

_PARENTS_SEARCH = 10


def get_vcs_info(path=pathlib.Path.cwd(), parent_search: int = 0) -> PyVcsInfo:
    try:
        repo = Repo(path)
    except InvalidGitRepositoryError:
        if parent_search > _PARENTS_SEARCH:
            LOGGER.error("Could not find Git Repo")
            return PyVcsInfo("NoSHAProvided", "NoBranchFound", None)
        else:
            return get_vcs_info(path.parent, parent_search + 1)
    if repo.is_dirty():
        LOGGER.info(
            "Current Repo is dirty - this hash may not represent the current working state"
        )
    remote = next(map(lambda x: git_parse(str(x.url)), repo.remotes), None)
    try:
        return PyVcsInfo(
            sha=str(repo.commit("HEAD")),
            branch=repo.active_branch.name,
            remote=PyRemoteRepository(remote.resource, remote.name, remote.owner)
            if remote
            else None,
        )
    except BadName:
        return PyVcsInfo(
            sha="DirtyHashState",
            branch=repo.active_branch.name,
            remote=PyRemoteRepository(remote.resource, remote.name, remote.owner)
            if remote
            else None,
        )
    except TypeError:
        # Detached HEAD State; https://github.com/gitpython-developers/GitPython/issues/633
        return PyVcsInfo(
            sha="DirtyHashState",
            branch="DetachedHead",
            remote=PyRemoteRepository(remote.resource, remote.name, remote.owner)
            if remote
            else None,
        )
