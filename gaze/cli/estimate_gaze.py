from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from ..estimators import available_estimators, available_variants
from ..core.pipeline import estimate_gaze_video


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gaze",
        description=(
            "Gaze augmentation stage (gaze-lib): attach gaze (yaw/pitch + gaze_vec) "
            "to detections using face boxes already present in the input JSON.\n\n"
            "Notes:\n"
            "- This tool does NOT run a face detector.\n"
            "- Nothing is written unless you enable --json/--frames/--save-video.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    p.add_argument("--json-in", required=False, type=Path, help="Path to upstream JSON (det/track/face/unknown accepted).")
    p.add_argument("--video", required=False, type=Path, help="Path to original video used for the upstream JSON.")

    # Discovery
    p.add_argument("--list-estimators", action="store_true", help="List available gaze estimator backends and exit.")
    p.add_argument("--list-variants", action="store_true", help="List supported variants for --estimator and exit.")
    p.add_argument("--estimator", default="l2cs", help="Estimator backend name.")
    p.add_argument("--variant", default=None, help="Named variant for the estimator backend. For l2cs, this selects the backbone (e.g., resnet50).")

    # Backend config
    p.add_argument("--weights", type=Path, default=None, help="Path to estimator weights (required for most backends).")
    p.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto|cpu|mps|cuda|cuda:0|0|1...",
    )
    p.add_argument(
        "--expand-face",
        type=float,
        default=0.0,
        help="Expand each face box by this fraction before cropping (e.g. 0.25 => +25%%).",
    )

    # Association / filtering
    p.add_argument(
        "--associate-classes",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Class IDs eligible for gaze attachment. "
            "If omitted, the tool tries to infer class_name=='person'; "
            "if not found, all classes are eligible."
        ),
    )
    p.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="If set, use this face index per detection; otherwise, choose best-score face.",
    )

    # Origin behavior
    p.add_argument(
        "--kpt-origin",
        nargs="*",
        type=int,
        default=None,
        help="Keypoint indices to compute gaze origin (mean of selected points).",
    )
    p.add_argument(
        "--kpt-conf",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold used with --kpt-origin.",
    )
    p.add_argument(
        "--fallback",
        action="store_true",
        help=(
            "If set, allow fallback to face/detection box center when keypoint-origin "
            "is unavailable; otherwise detections without keypoint-origin are skipped."
        ),
    )

    # Artifacts (opt-in)
    p.add_argument("--json", dest="save_json_flag", action="store_true", help="Write augmented JSON to <run>/gaze.json.")
    p.add_argument("--frames", dest="save_frames", action="store_true", help="Save annotated frames under <run>/frames/.")
    p.add_argument(
        "--save-video",
        nargs="?",
        const="annotated.mp4",
        default=None,
        help="Save annotated video under <run>/ (optionally pass a filename).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("out"), help="Output root used only when saving artifacts.")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name under --out-dir. If omitted, outputs go directly under --out-dir.",
    )
    p.add_argument("--fourcc", type=str, default="mp4v", help="FourCC codec for saved video.")
    p.add_argument("--display", action="store_true", help="Show live annotated preview (ESC to stop).")

    # UX
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = _build_argparser()
    args = p.parse_args(argv)

    if args.list_estimators:
        for name in available_estimators():
            print(name)
        return

    if args.list_variants:
        vs = available_variants(args.estimator)
        for v in vs:
            print(v)
        return

    # Validate required args for a real run
    if args.json_in is None or args.video is None:
        p.error("--json-in and --video are required unless using --list-estimators/--list-variants")

    if args.weights is None:
        p.error("--weights is required for running gaze estimation")

    res = estimate_gaze_video(
        json_in=args.json_in,
        video=args.video,
        estimator=args.estimator,
        variant=args.variant,
        weights=args.weights,
        device=args.device,
        expand_face=args.expand_face,
        associate_class_ids=args.associate_classes,
        face_index=args.face_index,
        kpt_origin=args.kpt_origin,
        kpt_conf=args.kpt_conf,
        fallback=args.fallback,
        save_json_flag=args.save_json_flag,
        save_frames=args.save_frames,
        save_video=args.save_video,
        out_dir=args.out_dir,
        run_name=args.run_name,
        fourcc=args.fourcc,
        display=args.display,
        no_progress=args.no_progress,
    )

    print(json.dumps(res.stats, indent=2))


if __name__ == "__main__":
    main()