from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# -------------------------
# JSON I/O
# -------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------------
# Video I/O
# -------------------------

class VideoReader:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 0 else 30.0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        *,
        fourcc: str = "mp4v",
    ):
        self.path = str(path)
        self._size = (int(size[0]), int(size[1]))
        self._fps = float(fps)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(self.path, code, self._fps, self._size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            frame = cv2.resize(frame, self._size)
        self.writer.write(frame)

    def release(self) -> None:
        try:
            self.writer.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False