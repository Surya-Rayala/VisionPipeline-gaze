from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from .base import GazeEstimator


@dataclass
class L2CSEstimator(GazeEstimator):
    """Thin wrapper around L2CS-Net for face-box-driven inference.

    Notes:
      - Never runs a face detector. You must provide face boxes per frame.
      - Accepts BGR frames; internally converts to RGB for L2CS Pipeline.
      - Returns pitch/yaw arrays (radians), shaped (N,).
    """

    weights: Path
    arch: str = "ResNet50"
    device: Union[str, torch.device] = "cpu"
    expand_face: float = 0.0  # fractional expansion: 0.25 => +25%

    # internal (initialized in __post_init__)
    _pipe: object | None = None
    _batch_buf: np.ndarray | None = None
    _batch_capacity: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Heavy import only when backend is constructed
        from l2cs import Pipeline as L2CSPipeline  # type: ignore

        self._pipe = L2CSPipeline(
            weights=Path(self.weights),
            arch=self.arch,
            device=self.device,
            include_detector=False,
        )

        # Ensure eval mode if accessible
        try:
            self._pipe.model.eval()  # type: ignore[attr-defined]
        except Exception:
            pass

        # Patch for upstream Pipeline when include_detector=False:
        # some versions define softmax/idx_tensor only when detector enabled.
        try:
            import torch.nn as nn

            if not hasattr(self._pipe, "softmax"):
                setattr(self._pipe, "softmax", nn.Softmax(dim=1))
            if not hasattr(self._pipe, "idx_tensor"):
                setattr(self._pipe, "idx_tensor", torch.arange(90, dtype=torch.float32, device=self.device))
        except Exception:
            # If patching fails, predict_gaze may fail; let it raise later.
            pass

    def predict_on_faces(
        self,
        frame_bgr: np.ndarray,
        face_boxes: Sequence[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 ndarray (BGR)")

        if self._pipe is None:
            raise RuntimeError("L2CSEstimator is not initialized")

        H, W = frame_bgr.shape[:2]
        if not face_boxes:
            return (np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Validate + expand + clamp boxes
        valid: List[Tuple[int, int, int, int]] = []
        f = float(self.expand_face or 0.0)

        for box in face_boxes:
            if not box or len(box) < 4:
                continue

            x1f, y1f, x2f, y2f = [float(v) for v in box[:4]]
            if x2f < x1f:
                x1f, x2f = x2f, x1f
            if y2f < y1f:
                y1f, y2f = y2f, y1f

            wf = x2f - x1f
            hf = y2f - y1f
            if wf <= 1.0 or hf <= 1.0:
                continue

            if f > 0.0:
                cx = 0.5 * (x1f + x2f)
                cy = 0.5 * (y1f + y2f)
                wf *= (1.0 + f)
                hf *= (1.0 + f)
                x1f = cx - 0.5 * wf
                x2f = cx + 0.5 * wf
                y1f = cy - 0.5 * hf
                y2f = cy + 0.5 * hf

            x1 = max(0, min(W, int(round(x1f))))
            x2 = max(0, min(W, int(round(x2f))))
            y1 = max(0, min(H, int(round(y1f))))
            y2 = max(0, min(H, int(round(y2f))))
            if x2 <= x1 or y2 <= y1:
                continue

            valid.append((x1, y1, x2, y2))

        if not valid:
            return (np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))

        # Ensure reusable batch buffer
        need = len(valid)
        if self._batch_buf is None or self._batch_capacity < need:
            self._batch_buf = np.empty((need, 224, 224, 3), dtype=np.uint8)
            self._batch_capacity = need

        # Fill buffer (RGB chips)
        buf = self._batch_buf
        for i, (x1, y1, x2, y2) in enumerate(valid):
            chip = frame_rgb[y1:y2, x1:x2]
            # cv2.resize requires non-empty; guaranteed by checks above
            buf[i] = cv2.resize(chip, (224, 224), interpolation=cv2.INTER_LINEAR)

        batch = buf[:need]

        with torch.inference_mode():
            yaw, pitch = self._pipe.predict_gaze(batch)  # type: ignore[attr-defined]

        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        yaw = np.asarray(yaw, dtype=np.float32).reshape(-1)
        return pitch, yaw


def build_l2cs_estimator(
    *,
    weights: Path | str,
    arch: str = "ResNet50",
    device: Union[str, torch.device] = "cpu",
    expand_face: float = 0.0,
) -> L2CSEstimator:
    """Factory used by the registry."""
    return L2CSEstimator(weights=Path(weights), arch=arch, device=device, expand_face=float(expand_face))