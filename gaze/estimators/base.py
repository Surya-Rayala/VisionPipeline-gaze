from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np


class GazeEstimator(Protocol):
    """Interface for gaze backends.

    Backends must accept a BGR frame and a list of face boxes (xyxy) and return:
      - pitch: (N,) radians
      - yaw:   (N,) radians
    """

    def predict_on_faces(
        self,
        frame_bgr: np.ndarray,
        face_boxes: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class EstimatorSpec:
    """Small descriptor used for listing/backends/metadata."""

    name: str
    variants: Tuple[str, ...] = ()
    description: str = ""