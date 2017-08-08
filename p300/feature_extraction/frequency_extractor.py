#! coding: utf-8
"""Feature Extractors."""
from scipy.signal import welch
import numpy as np
from . import BaseTransformer


class FrequencyExtractor(BaseTransformer):
    """Extractor of plain frequencies."""

    def __init__(self, min_freq=1, max_freq=20):
        """Constructor.

        Parameters:
        ----------

        passband: tuple of floats
            Passband to be returned
        """
        self.min_freq = min_freq
        self.max_freq = max_freq

    def get_feature_names(self):
        """Return name for features."""
        channel_names = "AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4".split(",")

        return ["{}_{}hz".format(ch_name, str(f)) for ch_name in channel_names for f in self.freqs]

    def transform(self, x, y=None):
        """Transform signal.

        Parameters:
        ----------

        x: np.array of channels
            array of nchannels x samples
        """
        self.freqs, magnitudes = welch(x, fs=128, nperseg=100)

        # Remove frequencies not wanted
        max_idx = np.argmax(self.freqs > self.max_freq)  # No more than x hz
        min_idx = np.argmin(self.freqs < self.min_freq)

        self.freqs = self.freqs[min_idx:max_idx]
        # Remove out-of-band frequencies
        magnitudes = magnitudes[:, :, min_idx:max_idx]
        # Reshape it into a 1-d array
        magnitudes = magnitudes.reshape(magnitudes.shape[0], -1)

        return magnitudes
