from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseTextDetector(ABC):
    """Interface for text-region detectors (returns quadrilaterals)."""

    name = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Return a list of quads (4x2 int arrays) around text areas."""

    def close(self) -> None:  # pragma: no cover
        pass
