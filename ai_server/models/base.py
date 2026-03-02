"""Abstract model interface — all models implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Base class for all AI models in the trading system."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """Return a probability 0.0 – 1.0 for the given feature input."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
