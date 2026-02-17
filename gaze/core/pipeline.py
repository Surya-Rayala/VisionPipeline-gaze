from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..estimators import get_gaze_estimator
from ..viz import draw_frame
from .io import VideoReader, VideoWriter, load_json, save_json
from .normalize import autodetect_associate_class_ids, normalize_payload
from .result import GazeResult
from .schema import (
    K_BBOX,
    K_CLASS_ID,
    K_CLASS_NAME,
    K_DETECTIONS,
    K_FACE_IND,
    K_FACES,
    K_FRAME_INDEX,
    K_FRAMES,
    K_GAZE,
    K_GAZE_AUGMENT,
    K_GAZE_ORIGIN,
    K_GAZE_ORIGIN_SRC,
    K_GAZE_VEC,
    K_PARENT_SCHEMA_VERSION,
    K_PITCH,
    K_SCHEMA_VERSION,
    K_SCORE,
    K_KEYPOINTS,
    K_YAW,
    angles_to_vec,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_xyxy_int(box: Sequence[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _pick_face_index(faces: List[Dict[str, Any]], face_index: Optional[int]) -> Optional[int]:
    if not faces:
        return None
    if face_index is not None and 0 <= face_index < len(faces):
        return int(face_index)
    # else best by score
    best_i = None
    best_s = -1.0
    for i, f in enumerate(faces):
        if not isinstance(f, dict):
            continue
        try:
            s = float(f.get(K_SCORE, 0.0))
        except Exception:
            s = 0.0
        if s > best_s:
            best_s = s
            best_i = i
    return int(best_i) if best_i is not None else 0


def _compute_origin_from_keypoints(
    det: Dict[str, Any],
    *,
    kpt_origin: Optional[Sequence[int]],
    kpt_conf: float,
) -> Optional[List[float]]:
    if not kpt_origin:
        return None
    pts = det.get(K_KEYPOINTS)
    if not isinstance(pts, list) or not pts:
        return None
    valid: List[List[float]] = []
    for idx in kpt_origin:
        if not isinstance(idx, int):
            continue
        if 0 <= idx < len(pts):
            pt = pts[idx]
            if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                try:
                    if float(pt[2]) >= float(kpt_conf):
                        valid.append([float(pt[0]), float(pt[1])])
                except Exception:
                    continue
    if not valid:
        return None
    arr = np.asarray(valid, dtype=float)
    mean = arr.mean(axis=0)
    return [float(mean[0]), float(mean[1])]


def _compute_origin_fallback(det: Dict[str, Any], face_box: Optional[Sequence[float]]) -> Optional[List[float]]:
    # Prefer face box center if available, else detection box center
    if face_box is not None and len(face_box) >= 4:
        x1, y1, x2, y2 = _ensure_xyxy_int(face_box)
        return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
    pb = det.get(K_BBOX)
    if pb is not None and isinstance(pb, (list, tuple)) and len(pb) >= 4:
        x1, y1, x2, y2 = _ensure_xyxy_int(pb)
        return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
    return None


def _resolve_out_path(run_dir: Path, maybe_name: Optional[str], default_name: str) -> Path:
    if not maybe_name:
        return run_dir / default_name
    p = Path(maybe_name)
    if p.is_absolute():
        return p
    return run_dir / p


@dataclass
class RunConfig:
    # Inputs
    json_in: Path
    video: Path

    # Backend
    estimator: str = "l2cs"
    variant: Optional[str] = None
    weights: Path | None = None
    device: str = "auto"
    expand_face: float = 0.0

    # Association / selection
    associate_class_ids: Optional[List[int]] = None
    face_index: Optional[int] = None

    # Origin behavior
    kpt_origin: Optional[List[int]] = None
    kpt_conf: float = 0.3
    fallback: bool = False  # allow box center fallback when keypoint origin missing

    # Output (opt-in)
    save_json_flag: bool = False
    save_frames: bool = False
    save_video: Optional[str] = None
    out_dir: Path = Path("out")
    run_name: Optional[str] = None
    fourcc: str = "mp4v"
    display: bool = False
    no_progress: bool = False


def estimate_gaze_video(**kwargs: Any) -> GazeResult:
    """Main library API: run gaze augmentation on video + upstream JSON.

    Returns a GazeResult(payload, paths, stats).
    """
    cfg = RunConfig(**kwargs)  # type: ignore[arg-type]
    if cfg.weights is None:
        raise ValueError("weights is required for estimator backends (e.g., L2CS).")

    # Load + normalize
    raw = load_json(cfg.json_in)
    norm = normalize_payload(raw)

    # Determine associate_class_ids
    associate_ids = cfg.associate_class_ids
    if associate_ids is None:
        inferred = autodetect_associate_class_ids(norm.payload)
        associate_ids = inferred  # may be None (meaning allow all)
    # None => all classes eligible

    # Warn once if user requested keypoint origin but none exist anywhere
    warned_no_kpts = False
    if cfg.kpt_origin and not norm.has_any_keypoints:
        warnings.warn(
            "kpt_origin was provided, but no keypoints were found in the input JSON. "
            "Keypoint-origin will be skipped; use --fallback to allow box-center origin.",
            RuntimeWarning,
        )
        warned_no_kpts = True

    # Construct estimator
    est = get_gaze_estimator(
        estimator=cfg.estimator,
        variant=cfg.variant,
        weights=cfg.weights,
        device=cfg.device,
        expand_face=cfg.expand_face,
    )

    # Decide whether we need to create output dirs (detect-face style)
    saving_enabled = bool(cfg.save_json_flag or cfg.save_frames or cfg.save_video)
    run_dir: Optional[Path] = None
    paths: Dict[str, Path] = {}
    writer: Optional[VideoWriter] = None
    frames_dir: Optional[Path] = None
    out_json_path: Optional[Path] = None
    out_video_path: Optional[Path] = None

    if saving_enabled:
        out_dir = Path(cfg.out_dir)
        if cfg.run_name and str(cfg.run_name).strip():
            run_dir = out_dir / str(cfg.run_name)
        else:
            run_dir = out_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        if cfg.save_json_flag:
            out_json_path = _resolve_out_path(run_dir, "gaze.json", "gaze.json")
            # If you want the default name to be "faces.json" like old code, change above.
            paths["json"] = out_json_path

        if cfg.save_frames:
            frames_dir = run_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            paths["frames_dir"] = frames_dir

        if cfg.save_video:
            out_video_path = _resolve_out_path(run_dir, cfg.save_video, "annotated.mp4")
            paths["video"] = out_video_path

    # Video IO
    reader = VideoReader(cfg.video)
    if out_video_path is not None:
        writer = VideoWriter(
            str(out_video_path),
            fps=reader.fps,
            size=(reader.width, reader.height),
            fourcc=cfg.fourcc,
        )

    # Add top-level metadata block (before frames)
    parent_sv = norm.schema_version
    norm.payload.setdefault(K_SCHEMA_VERSION, parent_sv if parent_sv else "unknown")
    norm.payload[K_GAZE_AUGMENT] = {
        "version": "gaze-v1",
        K_PARENT_SCHEMA_VERSION: parent_sv,
        "estimator": {
            "name": cfg.estimator,
            "variant": cfg.variant,
            "weights": str(cfg.weights),
            "device": cfg.device,
            "expand_face": float(cfg.expand_face),
        },
        "association": {
            "associate_class_ids": associate_ids,
            "face_index": cfg.face_index,
            "kpt_origin": cfg.kpt_origin,
            "kpt_conf": float(cfg.kpt_conf),
            "fallback": bool(cfg.fallback),
        },
        "run": {
            "created_utc": _utc_now_iso(),
            "tool": "gaze",
            "tool_version": norm.payload.get("__tool_version__", "unknown"),
        },
    }

    frames = norm.payload.get(K_FRAMES, [])
    total_frames = 0
    total_candidates = 0
    total_attached = 0

    # Optional progress
    pbar = None
    if not cfg.no_progress:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(frames), desc="gaze")
        except Exception:
            pbar = None

    try:
        for fidx, fobj in enumerate(frames):
            ok, frame = reader.read()
            if not ok:
                break
            total_frames += 1

            if not isinstance(fobj, dict):
                if pbar:
                    pbar.update(1)
                continue

            dets = fobj.get(K_DETECTIONS)
            if not isinstance(dets, list) or not dets:
                # still allow display/video passthrough
                if writer is not None:
                    writer.write(frame)
                if cfg.display:
                    cv2.imshow("gaze", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        cfg.display = False
                if frames_dir is not None:
                    cv2.imwrite(str(frames_dir / f"{fidx:06d}.jpg"), frame)
                if pbar:
                    pbar.update(1)
                continue

            boxes: List[Tuple[int, int, int, int]] = []
            mapping: List[Tuple[int, int]] = []  # (det_index, face_index)

            # Build face-box batch per frame
            for di, det in enumerate(dets):
                if not isinstance(det, dict):
                    continue

                # class filter
                if associate_ids is not None:
                    try:
                        cid = int(det.get(K_CLASS_ID, -1))
                    except Exception:
                        cid = -1
                    if cid not in associate_ids:
                        continue

                faces = det.get(K_FACES)
                if not isinstance(faces, list) or not faces:
                    continue

                fi = _pick_face_index(faces, cfg.face_index)
                if fi is None or not (0 <= fi < len(faces)):
                    continue

                face_obj = faces[fi]
                if not isinstance(face_obj, dict):
                    continue
                fb = face_obj.get(K_BBOX)
                if not isinstance(fb, (list, tuple)) or len(fb) < 4:
                    continue

                x1, y1, x2, y2 = _ensure_xyxy_int(fb)
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append((x1, y1, x2, y2))
                mapping.append((di, fi))
                total_candidates += 1

            # Run estimator
            if boxes:
                pitch, yaw = est.predict_on_faces(frame, boxes)
                # Ensure alignment
                n = min(len(mapping), int(pitch.shape[0]), int(yaw.shape[0]))

                for i in range(n):
                    di, fi = mapping[i]
                    det = dets[di]
                    face_box = boxes[i]

                    # origin: keypoints if requested, else optional fallback
                    origin = _compute_origin_from_keypoints(det, kpt_origin=cfg.kpt_origin, kpt_conf=cfg.kpt_conf)
                    origin_src = None

                    if origin is not None:
                        origin_src = "kpt"
                    else:
                        # Warn once per run if keypoints exist nowhere is already done;
                        # but also if keypoints exist in some frames but not this det, we skip silently unless fallback.
                        if cfg.fallback:
                            origin = _compute_origin_fallback(det, face_box)
                            if origin is not None:
                                origin_src = "box"

                    # If kpt-origin requested but missing and fallback disabled => skip attach
                    if cfg.kpt_origin and origin is None and not cfg.fallback:
                        continue

                    yv = float(yaw[i])
                    pv = float(pitch[i])
                    gx, gy, gz = angles_to_vec(yv, pv)

                    det[K_GAZE] = {
                        K_YAW: yv,
                        K_PITCH: pv,
                        K_GAZE_VEC: [gx, gy, gz],
                        K_FACE_IND: int(fi),
                    }
                    if origin is not None and origin_src is not None:
                        det[K_GAZE][K_GAZE_ORIGIN] = origin
                        det[K_GAZE][K_GAZE_ORIGIN_SRC] = origin_src

                    total_attached += 1

            # Visualization if needed (only when display/saving video/frames)
            need_viz = bool(cfg.display or writer is not None or frames_dir is not None)
            out_img = frame
            if need_viz:
                vis = frame.copy()
                out_img = draw_frame(vis, dets)

            if writer is not None:
                writer.write(out_img)

            if cfg.display:
                cv2.imshow("gaze", out_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    cfg.display = False

            if frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"{fidx:06d}.jpg"), out_img)

            if pbar:
                pbar.update(1)

        # Save JSON if enabled
        if cfg.save_json_flag and out_json_path is not None:
            # Re-order top-level keys so `gaze_augment` appears before `frames` in the saved JSON.
            # (Python preserves insertion order when dumping dicts.)
            payload = norm.payload
            gaze_aug = payload.get(K_GAZE_AUGMENT)
            frames_val = payload.get(K_FRAMES)

            ordered: Dict[str, Any] = {}
            for k, v in payload.items():
                if k in (K_GAZE_AUGMENT, K_FRAMES):
                    continue
                ordered[k] = v

            if gaze_aug is not None:
                ordered[K_GAZE_AUGMENT] = gaze_aug
            if frames_val is not None:
                ordered[K_FRAMES] = frames_val

            norm.payload = ordered
            save_json(norm.payload, out_json_path)

    finally:
        if writer is not None:
            writer.release()
        reader.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    stats = {
        "frames_processed": total_frames,
        "candidates": total_candidates,
        "gaze_attached": total_attached,
        "schema_version_in": parent_sv,
        "associate_class_ids": associate_ids,
        "saving_enabled": saving_enabled,
        "out_dir": str(cfg.out_dir),
        "run_name": cfg.run_name,
    }
    if out_json_path is not None:
        stats["out_json"] = str(out_json_path)
    if out_video_path is not None:
        stats["out_video"] = str(out_video_path)
    if frames_dir is not None:
        stats["frames_dir"] = str(frames_dir)

    return GazeResult(payload=norm.payload, paths=paths, stats=stats)