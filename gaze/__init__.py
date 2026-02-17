"""
gaze-lib (module: gaze)

Modular gaze augmentation toolkit that attaches gaze estimates to detections
using face boxes already present in the input JSON.

- Does NOT run a face detector.
- Accepts upstream JSON with frames/detections and optional face boxes.
- Returns augmented payload in-memory; writes artifacts only when enabled.
"""
from __future__ import annotations

from typing import Any

from .estimators import available_estimators, available_variants

__all__ = [
    "estimate_gaze_video",
    "get_gaze_estimator",
    "available_estimators",
    "available_variants",
    "__version__",
]

__version__ = "0.1.0"


def estimate_gaze_video(*args: Any, **kwargs: Any):
    """Lazy proxy to :func:`gaze.core.pipeline.estimate_gaze_video`."""
    from .core.pipeline import estimate_gaze_video as _impl

    return _impl(*args, **kwargs)


def get_gaze_estimator(*args: Any, **kwargs: Any):
    """Lazy proxy to :func:`gaze.estimators.registry.get_gaze_estimator`."""
    from .estimators import get_gaze_estimator as _impl

    return _impl(*args, **kwargs)