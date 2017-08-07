#! coding: utf-8
"""Feature Extractors."""
from sklearn.base import TransformerMixin
from scipy import signal


class LoadArray(TransformerMixin):
    """Load numpy array transformer."""
    def transform(self, x, y=None):
        """Transform method."""
        pass


class DownsampleExtractor(TransformerMixin):
    """Downsampler."""

    def transform(self, x, y=None):
        """Transform method."""
        pass


class FrequencyExtractor(TransformerMixin):
    """Extractor of plain frequencies."""

    def transform(self, x, y=None):
        """Transform signal."""
        self.freqs, magnitudes = signal.welch(x, fs=128, nperseg=64)

        return magnitudes
