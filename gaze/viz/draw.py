from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.schema import (
    K_BBOX,
    K_CLASS_ID,
    K_CLASS_NAME,
    K_DETECTIONS,
    K_FACE_IND,
    K_FACES,
    K_GAZE,
    K_GAZE_ORIGIN,
    K_KEYPOINTS,
    K_LANDMARKS,
    K_PITCH,
    K_SCORE,
    K_SEGMENTS,
    K_YAW,
    angles_to_vec,
)

# -------------------------
# Color + tiny draw utils
# -------------------------

@lru_cache(maxsize=1024)
def _hash_to_color(key: str, s: float = 0.9, v: float = 0.95) -> Tuple[int, int, int]:
    import hashlib

    if key is None:
        key = "_"
    if not isinstance(key, str):
        key = str(key)
    hval = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    hue = (hval % 360) / 360.0

    i = int(hue * 6)
    f = hue * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(b * 255), int(g * 255), int(r * 255)


def _lighten(color: Tuple[int, int, int], amt: float = 0.35) -> Tuple[int, int, int]:
    c = np.array(color, dtype=np.float32)
    w = np.array([255, 255, 255], dtype=np.float32)
    out = (1 - amt) * c + amt * w
    return int(out[0]), int(out[1]), int(out[2])


def _darken(color: Tuple[int, int, int], amt: float = 0.35) -> Tuple[int, int, int]:
    c = np.array(color, dtype=np.float32)
    out = (1 - amt) * c
    return int(out[0]), int(out[1]), int(out[2])


def _legible_text_color(bg_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bg_bgr
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if y > 160 else (255, 255, 255)


def _clamp_point(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


# -------------------------
# Primitive drawing
# -------------------------

def draw_bbox(img: np.ndarray, box: Sequence[float], color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    h, w = img.shape[:2]
    x1, y1 = _clamp_point(x1, y1, w, h)
    x2, y2 = _clamp_point(x2, y2, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_label_block(
    img: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    color: Tuple[int, int, int],
    *,
    font_scale: float | None = None,
) -> None:
    ts = font_scale if font_scale is not None else max(0.4, min(1.6, img.shape[0] / 1080.0))
    tf = max(1, int(round(ts * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts, tf)
    th = th + 4

    y_top = y1 - th
    if y_top < 0:
        y_top = y1
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y_top + th), color, -1)
        cv2.putText(img, text, (x1 + 3, y_top + th - 4), cv2.FONT_HERSHEY_SIMPLEX, ts, _legible_text_color(color), tf, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, ts, _legible_text_color(color), tf, cv2.LINE_AA)


def draw_keypoints(img: np.ndarray, kpts: Sequence[Sequence[float]], color: Tuple[int, int, int], radius: int = 3) -> None:
    for item in kpts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        x, y = int(item[0]), int(item[1])
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_face_landmarks(img: np.ndarray, pts: Iterable[Iterable[float]], color: Tuple[int, int, int]) -> None:
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x = int(round(float(p[0]))); y = int(round(float(p[1])))
        h, w = img.shape[:2]
        x, y = _clamp_point(x, y, w, h)
        cv2.line(img, (x - 2, y), (x + 2, y), _lighten(color, 0.15), 1, cv2.LINE_AA)
        cv2.line(img, (x, y - 2), (x, y + 2), _lighten(color, 0.15), 1, cv2.LINE_AA)


def _as_polygon(seg: Any) -> Optional[np.ndarray]:
    """Try to interpret seg as a polygon Nx2 array."""
    if seg is None:
        return None
    # accept list of [x,y] points
    if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)) and len(seg[0]) >= 2:
        try:
            pts = np.asarray(seg, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] >= 2:
                return pts[:, :2]
        except Exception:
            return None
    return None


def draw_polygon(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.2, thickness: int = 2) -> None:
    if pts is None or len(pts) < 3:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], _lighten(color, 0.4))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], isClosed=True, color=_darken(color, 0.3), thickness=thickness, lineType=cv2.LINE_AA)


def draw_gaze_vector(
    img: np.ndarray,
    face_box: Sequence[float],
    yaw: float,
    pitch: float,
    color: Tuple[int, int, int],
    *,
    origin: Optional[Sequence[float]] = None,
) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in face_box[:4]]
    h, w = img.shape[:2]
    x1, y1 = _clamp_point(x1, y1, w, h)
    x2, y2 = _clamp_point(x2, y2, w, h)

    if origin is not None and len(origin) >= 2:
        ox = int(round(float(origin[0])))
        oy = int(round(float(origin[1])))
        ox, oy = _clamp_point(ox, oy, w, h)
        p1 = (ox, oy)
        length = max(8, int(1 * max(1, x2 - x1)))
    else:
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        p1 = (cx, cy)
        length = max(8, int(1 * max(1, x2 - x1)))

    gx, gy, _gz = angles_to_vec(float(yaw), float(pitch))
    dx = gx * length
    dy = gy * length
    p2 = (int(round(p1[0] + dx)), int(round(p1[1] + dy)))

    cv2.arrowedLine(img, p1, p2, (0, 0, 0), 4, cv2.LINE_AA, tipLength=0.20)
    cv2.arrowedLine(img, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.20)
    cv2.circle(img, p1, 3, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, p1, 2, color, -1, cv2.LINE_AA)


# -------------------------
# Public frame composer
# -------------------------

def draw_frame(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    *,
    show_boxes: bool = True,
    show_faces: bool = True,
    show_kpts: bool = True,
    show_segs: bool = True,
    show_gaze: bool = True,
) -> np.ndarray:
    if img is None:
        return img

    for det in detections:
        if not isinstance(det, dict):
            continue

        sid = (
            str(det.get("gallery_id") or "")
            or str(det.get("track_id") or "")
            or f"det{det.get('det_ind', '')}"
            or str(det.get(K_CLASS_ID, ""))
        )
        base = _hash_to_color(sid)
        face_c = _lighten(base, 0.25)
        gaze_c = _darken(base, 0.15)

        # segments
        if show_segs:
            seg = det.get(K_SEGMENTS) or det.get("seg") or det.get("segments")
            pts = _as_polygon(seg)
            if pts is not None:
                draw_polygon(img, pts, base, alpha=0.2, thickness=2)

        # person bbox + label
        if show_boxes and det.get(K_BBOX) is not None:
            draw_bbox(img, det[K_BBOX], base, thickness=2)
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in det[K_BBOX][:4]]
            except Exception:
                x1 = y1 = 0
            label_parts: List[str] = []
            if det.get(K_CLASS_NAME):
                label_parts.append(str(det[K_CLASS_NAME]))
            if det.get("track_id"):
                label_parts.append(str(det.get("track_id")))
            if det.get(K_SCORE) is not None:
                try:
                    label_parts.append(f"{float(det[K_SCORE]):.2f}")
                except Exception:
                    pass
            if label_parts:
                draw_label_block(img, x1, y1, " ".join(label_parts), base)

        # keypoints
        if show_kpts:
            kps = det.get(K_KEYPOINTS)
            if isinstance(kps, list) and kps:
                draw_keypoints(img, kps, base, radius=3)

        # faces + landmarks
        if show_faces:
            faces = det.get(K_FACES)
            if isinstance(faces, list):
                for face in faces:
                    if not isinstance(face, dict):
                        continue
                    box = face.get(K_BBOX)
                    if box is not None:
                        draw_bbox(img, box, face_c, thickness=1)
                        try:
                            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
                        except Exception:
                            x1 = y1 = 0
                        flabel = "face"
                        if face.get(K_SCORE) is not None:
                            try:
                                flabel = f"face {float(face[K_SCORE]):.2f}"
                            except Exception:
                                pass
                        draw_label_block(img, x1, y1, flabel, face_c)
                    lm = face.get(K_LANDMARKS)
                    if isinstance(lm, list) and lm:
                        draw_face_landmarks(img, lm, face_c)

        # gaze
        if show_gaze and isinstance(det.get(K_GAZE), dict):
            g = det[K_GAZE]
            yaw = float(g.get(K_YAW, 0.0))
            pitch = float(g.get(K_PITCH, 0.0))

            # pick anchor face box
            face_box = None
            fidx = -1
            if g.get(K_FACE_IND) is not None:
                try:
                    fidx = int(g.get(K_FACE_IND))
                except Exception:
                    fidx = -1

            faces = det.get(K_FACES)
            if fidx >= 0 and isinstance(faces, list) and fidx < len(faces) and isinstance(faces[fidx], dict):
                face_box = faces[fidx].get(K_BBOX)

            if face_box is None and isinstance(faces, list) and faces:
                if isinstance(faces[0], dict):
                    face_box = faces[0].get(K_BBOX)

            if face_box is None:
                face_box = det.get(K_BBOX)

            if face_box is not None:
                origin = None
                if isinstance(g.get(K_GAZE_ORIGIN), (list, tuple)) and len(g[K_GAZE_ORIGIN]) >= 2:
                    origin = g[K_GAZE_ORIGIN]
                draw_gaze_vector(img, face_box, yaw=yaw, pitch=pitch, color=gaze_c, origin=origin)

    return img