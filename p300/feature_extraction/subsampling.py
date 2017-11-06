#! coding: utf-8
"""Feature Extractors."""
from . import BaseTransformer
import numpy as np

class SubsamplingExtractor(BaseTransformer):
    """Subsampler."""

    def __init__(self, order=4):
        """Constructor.

        Parameters
        ----------

        order: int
            Indicates the order of the subsampling. Defaults to 4
        """
        self.order = order

    def get_feature_names(self):
        """Return name for features."""
        channel_names = "AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4"
        channel_names = channel_names.split(",")

        return [
            "{}_{}".format(ch_name, str(f)) for ch_name in channel_names
            for f in range(self.number_of_features)
        ]

    def transform(self, x, y=None):
        """Transform signal.

        Parameters:
        ----------

        x: np.array of channels
            array of nchannels x samples
        """
        sample_length = x.shape[-1]
        sampled_cumsum = x.cumsum(axis=-1)[..., (self.order-1)::self.order]

        avg_cumsum = np.concatenate([
                sampled_cumsum[..., 0][..., np.newaxis],
                np.diff(sampled_cumsum),
            ], axis=-1)

        avg_cumsum = avg_cumsum / self.order

        self.number_of_features = int(sample_length / self.order)

        return avg_cumsum.reshape(x.shape[0], -1)
