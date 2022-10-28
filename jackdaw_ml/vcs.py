__all__ = ["get_current_hash", "VCSHash"]

import logging
import pathlib
from typing import NamedTuple, List

from git import Repo, InvalidGitRepositoryError
from gitdb.exc import BadName
from urllib3.util import parse_url
from urllib3.util.url import Url

LOGGER = logging.getLogger(__name__)

_PARENTS_SEARCH = 10


class VCSHash(NamedTuple):
    hash: str
    remote: List[Url]


def get_current_hash(path=pathlib.Path.cwd(), parent_search: int = 0) -> VCSHash:
    try:
        repo = Repo(path)
    except InvalidGitRepositoryError:
        if parent_search > _PARENTS_SEARCH:
            raise ValueError("Could not find Git Repo")
        else:
            return get_current_hash(path.parent, parent_search + 1)
    if repo.is_dirty():
        LOGGER.warning(
            "Current Repo is dirty - this hash may not represent the current working state"
        )
    try:
        return VCSHash(
            str(repo.commit("HEAD")), list(map(lambda x: parse_url(str(x)), repo.remotes))
        )
    except BadName:
        return VCSHash("DirtyHashState", list(map(lambda x: parse_url(str(x)), repo.remotes)))
