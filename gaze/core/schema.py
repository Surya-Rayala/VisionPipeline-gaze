from __future__ import annotations

import math
from typing import Any, Iterable, List, Literal, Sequence, Tuple, TypedDict

# -------------------------
# Stable JSON keys (pipeline-compatible)
# -------------------------
K_SCHEMA_VERSION = "schema_version"
K_PARENT_SCHEMA_VERSION = "parent_schema_version"

K_VIDEO = "video"
K_FRAMES = "frames"
K_FRAME_INDEX = "frame_index"
K_DETECTIONS = "detections"

K_BBOX = "bbox"
K_SCORE = "score"
K_CLASS_ID = "class_id"
K_CLASS_NAME = "class_name"
K_KEYPOINTS = "keypoints"
K_SEGMENTS = "segments"  # keep plural here; normalize will be flexible

# Face entries (canonical output key is plural "faces"; normalizer accepts both)
K_FACES = "faces"
K_LANDMARKS = "landmarks"
K_FACE_IND = "face_ind"

# Gaze augmentation
K_GAZE_AUGMENT = "gaze_augment"
K_GAZE = "gaze"
K_YAW = "yaw"            # radians
K_PITCH = "pitch"        # radians
K_GAZE_VEC = "gaze_vec"  # [x,y,z]
K_GAZE_ORIGIN = "origin"  # [x,y] pixels
K_GAZE_ORIGIN_SRC = "origin_source"  # "kpt" | "box"

GazeOriginSource = Literal["kpt", "box"]

__all__ = [
    # keys
    "K_SCHEMA_VERSION",
    "K_PARENT_SCHEMA_VERSION",
    "K_VIDEO",
    "K_FRAMES",
    "K_FRAME_INDEX",
    "K_DETECTIONS",
    "K_BBOX",
    "K_SCORE",
    "K_CLASS_ID",
    "K_CLASS_NAME",
    "K_KEYPOINTS",
    "K_SEGMENTS",
    "K_FACES",
    "K_LANDMARKS",
    "K_FACE_IND",
    "K_GAZE_AUGMENT",
    "K_GAZE",
    "K_YAW",
    "K_PITCH",
    "K_GAZE_VEC",
    "K_GAZE_ORIGIN",
    "K_GAZE_ORIGIN_SRC",
    # types
    "FaceDet",
    "Gaze",
    "Detection",
    "Frame",
    "GazeOriginSource",
    # helpers
    "ensure_xyxy",
    "clip_box_to_image",
    "angles_to_vec",
]


# -------------------------
# Typed structures (lightweight; runtime not enforced)
# -------------------------

class FaceDet(TypedDict, total=False):
    bbox: List[float]              # [x1,y1,x2,y2]
    score: float
    landmarks: List[List[float]]   # [[x,y]...]
    face_ind: int


class Gaze(TypedDict, total=False):
    yaw: float
    pitch: float
    gaze_vec: List[float]          # [x,y,z]
    face_ind: int
    origin: List[float]            # [x,y]
    origin_source: GazeOriginSource


class Detection(TypedDict, total=False):
    bbox: List[float]
    score: float
    class_id: int
    class_name: str
    keypoints: List[List[float]]
    segments: Any
    faces: List[FaceDet]
    gaze: Gaze


class Frame(TypedDict, total=False):
    frame_index: int
    detections: List[Detection]


# -------------------------
# Geometry helpers
# -------------------------

def ensure_xyxy(box: Iterable[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box][:4]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clip_box_to_image(box: Iterable[float], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = ensure_xyxy(box)
    w, h = int(size[0]), int(size[1])
    x1i = max(0, min(w, int(round(x1))))
    y1i = max(0, min(h, int(round(y1))))
    x2i = max(0, min(w, int(round(x2))))
    y2i = max(0, min(h, int(round(y2))))
    return x1i, y1i, x2i, y2i


# -------------------------
# Angle → unit vector (L2CS convention)
# -------------------------

def angles_to_vec(yaw: float, pitch: float) -> Tuple[float, float, float]:
    """Convert yaw/pitch (radians) to a unit 3D gaze vector (x,y,z).

    Convention (L2CS-style):
      - yaw (left/right) affects x and z
      - pitch (up/down) affects y
      - Positive yaw = left turn, positive pitch = down
    """
    x = -math.sin(float(yaw)) * math.cos(float(pitch))
    y = -math.sin(float(pitch))
    z = -math.cos(float(yaw)) * math.cos(float(pitch))
    return float(x), float(y), float(z)