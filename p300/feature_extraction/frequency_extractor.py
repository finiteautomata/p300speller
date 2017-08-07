#! coding: utf-8
"""Feature Extractors."""
from scipy import signal
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
        return ["{}_hz".format(str(f)) for f in self.freqs]

    def transform(self, x, y=None):
        """Transform signal."""
        self.freqs, magnitudes = signal.welch(x, fs=128, nperseg=100)

        # Remove frequencies not wanted
        max_idx = np.argmax(self.freqs > self.max_freq)  # No more than x hz
        min_idx = np.argmin(self.freqs < self.min_freq)

        self.freqs = self.freqs[min_idx:max_idx]
        magnitudes = magnitudes[:, :, min_idx:max_idx]

        return magnitudes
