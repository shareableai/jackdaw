import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

from jackdaw_ml.saves import saves
from jackdaw_ml.loads import loads
