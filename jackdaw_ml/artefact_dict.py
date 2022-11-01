__all__ = ["ArtefactDict"]

from jackdaw_ml.artefact_decorator import artefacts

_SYSTEM_ITEMS = [
    "dumps",
    "loads",
    "__detectors__",
    "__artefact_slots__",
    "__artefact_children__",
    "__artefact_endpoint__",
    "__storage_location__",
]


class ArtefactDict(dict):
    __system_dict__ = {}
    """
    Special Form of default dict that allows dumps/loads methods
    """

    def __init__(self, storage_location=None, seq=None, **kwargs):
        super(ArtefactDict, self).__init__(seq, **kwargs)
        self = artefacts({}, storage_location=storage_location)(self)

    def __setattr__(self, key, value):
        if key in _SYSTEM_ITEMS:
            self.__system_dict__.__setitem__(key, value)
        else:
            self.__setitem__(key, value)

    def __getattr__(self, item):
        if item in _SYSTEM_ITEMS:
            if (value := self.__system_dict__.get(item)) is None:
                raise AttributeError(f"{item} not in {self}")
            else:
                return value
        else:
            if (value := self.get(item)) is None:
                raise AttributeError(f"{item} not in {self}")
            else:
                return value

    def keys(self):
        return [a for (a, _) in self.items()]

    def values(self):
        return [b for (_, b) in self.items()]

    def items(self):
        return [
            (key, value)
            for (key, value) in super(self).items()
            if key not in _SYSTEM_ITEMS
        ]

    def copy(self):
        inner_copy = ArtefactDict(super(self).copy())
        inner_copy.__system_dict__ = self.__system_dict__
        return inner_copy

    # TODO: Implement pop, popitem, setdefault, update, contains, eq,
    #   iter.
