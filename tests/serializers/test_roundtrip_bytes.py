from itertools import product

import pytest

from jackdaw.serializers import Serializable
from jackdaw.serializers.pickle import PickleSerializer
from tests.conftest import serializable_items

serializers = [PickleSerializer]


@pytest.mark.parametrize("item,serializer", product(serializable_items, serializers))
def test_roundtrip(item, serializer: Serializable):
    assert serializer.from_resource(None, serializer.to_resource(item)) == item
