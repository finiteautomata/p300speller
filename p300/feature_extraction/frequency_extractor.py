#! coding: utf-8
from sklearn.base import TransformerMixin
from scipy import signal

class FrequencyExtractor(TransformerMixin):
    """Extractor of plain frequencies."""

    def transform(self, x, y=None):
        """Transform signal."""
        freqs, magnitudes = signal.welch(x, fs=128, nperseg=100)

        return magnitudes
