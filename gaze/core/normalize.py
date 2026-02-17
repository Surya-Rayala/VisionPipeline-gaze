from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    K_CLASS_ID,
    K_CLASS_NAME,
    K_DETECTIONS,
    K_FACES,
    K_FRAMES,
    K_KEYPOINTS,
    K_SCHEMA_VERSION,
    K_BBOX,
)


@dataclass
class Normalized:
    """Normalized view over a loosely-structured detection JSON."""
    payload: Dict[str, Any]
    schema_version: str
    has_any_keypoints: bool
    has_any_faces: bool
    frames_count: int


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _normalize_schema_version(payload: Dict[str, Any]) -> str:
    sv = payload.get(K_SCHEMA_VERSION)
    if isinstance(sv, str) and sv.strip():
        return sv.strip()
    return "unknown"


def _coerce_bbox_xyxy(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1 = float(box[0]); y1 = float(box[1]); x2 = float(box[2]); y2 = float(box[3])
    except Exception:
        return None
    # normalize ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_payload(payload: Dict[str, Any]) -> Normalized:
    """Light normalization so downstream code can rely on containers.

    Guarantees:
      - payload["frames"] is a list
      - each frame has payload["detections"] list
      - each detection's "faces" (if present) is a list of dicts with valid bbox floats
    """
    schema_version = _normalize_schema_version(payload)

    frames = payload.get(K_FRAMES)
    if not isinstance(frames, list):
        payload[K_FRAMES] = []
        return Normalized(
            payload=payload,
            schema_version=schema_version,
            has_any_keypoints=False,
            has_any_faces=False,
            frames_count=0,
        )

    has_kpts = False
    has_faces = False

    for fobj in frames:
        if not isinstance(fobj, dict):
            continue

        dets = fobj.get(K_DETECTIONS)
        if not isinstance(dets, list):
            fobj[K_DETECTIONS] = []
            continue

        for det in dets:
            if not isinstance(det, dict):
                continue

            # keypoints existence (we don't validate shape here)
            if det.get(K_KEYPOINTS) is not None:
                if isinstance(det.get(K_KEYPOINTS), list):
                    has_kpts = True

            # Accept either legacy singular `face` or canonical plural `faces`.
            faces = det.get(K_FACES)
            if faces is None and "face" in det:
                faces = det.get("face")

            if isinstance(faces, list):
                cleaned: List[Dict[str, Any]] = []
                for face in faces:
                    if not isinstance(face, dict):
                        continue
                    box = face.get(K_BBOX)
                    if box is None and "bbox" in face:
                        box = face.get("bbox")
                    xyxy = _coerce_bbox_xyxy(box)
                    if xyxy is None:
                        continue
                    face[K_BBOX] = xyxy
                    cleaned.append(face)
                det[K_FACES] = cleaned  # canonicalize to "faces"
                # Remove legacy singular key if present
                if "face" in det:
                    try:
                        del det["face"]
                    except Exception:
                        pass
                if cleaned:
                    has_faces = True
            else:
                # If not a list, ensure it's absent or canonical empty list
                if K_FACES in det:
                    det[K_FACES] = []
                # Remove legacy singular key if present
                if "face" in det:
                    try:
                        del det["face"]
                    except Exception:
                        pass

    return Normalized(
        payload=payload,
        schema_version=schema_version,
        has_any_keypoints=has_kpts,
        has_any_faces=has_faces,
        frames_count=len(frames),
    )


def autodetect_associate_class_ids(payload: Dict[str, Any]) -> Optional[List[int]]:
    """Try to infer person-class IDs by class_name == 'person'.

    Returns:
      - [ids...] if found
      - None if not detectable
    """
    frames = payload.get(K_FRAMES)
    if not isinstance(frames, list):
        return None

    found: set[int] = set()
    for fobj in frames:
        if not isinstance(fobj, dict):
            continue
        dets = fobj.get(K_DETECTIONS)
        if not isinstance(dets, list):
            continue
        for det in dets:
            if not isinstance(det, dict):
                continue
            name = det.get(K_CLASS_NAME)
            if isinstance(name, str) and name.strip().lower() == "person":
                cid = det.get(K_CLASS_ID)
                try:
                    found.add(int(cid))
                except Exception:
                    continue

    if not found:
        return None
    return sorted(found)