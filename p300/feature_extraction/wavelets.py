#! coding: utf-8
"""Feature Extractors."""
from scipy import signal
import numpy as np
from . import BaseTransformer


class WaveletExtractor(BaseTransformer):
    """Extractor of plain frequencies."""

    def __init__(self, widths=np.arange(1, 10, 2)):
        """Constructor.

        Parameters:
        ----------
        widths: Array of numbers
            Widths used in cwt algorithm
        """
        self.widths = widths

    def get_feature_names(self):
        """Return name for features."""
        # TODO: remove this hardcoded stuff!
        channel_names = "AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4"
        channel_names = channel_names.split(",")

        def fname(chname, p, w):
            return "{}_p{}w{}".format(chname, p, w)

        return [
            fname(chname, p, w) for chname in channel_names
            # TODO: change this magic number
            for p in range(104)
            for w in self.widths
        ]

    def transform(self, X, y=None):
        """Transform signal.

        Parameters:
        ----------

        x: np.array of channels
            array of nchannels x samples
        """
        ret = []

        for instance in X:
            cwts = []
            # Instance is now an array of 14 channels x 104 points
            for channel in instance:
                cwt = signal.cwt(channel, signal.ricker, widths=self.widths)
                cwts.append(cwt.reshape(-1))

            features = np.concatenate(cwts)

            ret.append(features)

        ret = np.array(ret)
        return ret
