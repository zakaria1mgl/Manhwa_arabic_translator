from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.types import Quad


class OCREngineBase(ABC):
    """Common OCR interface. Returns dicts with english/confidence/quad."""

    name = "base"

    @abstractmethod
    def read(self, image_bgr: np.ndarray, quads: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        ...

    def close(self) -> None:  # pragma: no cover
        pass
