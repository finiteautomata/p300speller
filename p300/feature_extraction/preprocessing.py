"""Preprocessing of data."""
import numpy as np
from . import BaseTransformer


class LoadArray(BaseTransformer):
    """Load numpy array transformer."""

    def __init__(self, remove_sti=True):
        """Constructor.

        Parameters
        ----------

        remove_sti: Boolean (default true)
            Indicates whether to remove the Stimulus channel
        """
        self.remove_sti = remove_sti

    def transform(self, x, y=None):
        """Transform method."""
        ret = np.array([np.load(path) for path in x])

        if self.remove_sti:
            # TODO: Remove this hardcoded stuff
            ret = ret[:, :14, :]

        return ret
