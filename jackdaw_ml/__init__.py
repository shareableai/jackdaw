import logging
from logging import NullHandler

__name__ = "jackdaw_ml"
__version__ = "0.0.1"

logging.getLogger(__name__).addHandler(NullHandler())
