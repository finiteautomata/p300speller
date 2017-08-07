#! coding: utf-8
"""Base transformer."""
from sklearn.base import TransformerMixin


class BaseTransformer(TransformerMixin):
    """Clase base para todos nuestros transformers."""

    def fit(self, x, y=None):
        u"""Este método no hace nada, pero debe estar."""
        return self
