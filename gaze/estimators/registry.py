from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .base import EstimatorSpec, GazeEstimator
from .l2cs import build_l2cs_estimator


# -------------------------
# Registry (name -> spec + builder)
# -------------------------

@dataclass(frozen=True)
class _Entry:
    spec: EstimatorSpec
    builder: Callable[..., GazeEstimator]


_REGISTRY: Dict[str, _Entry] = {
    "l2cs": _Entry(
        spec=EstimatorSpec(
            name="l2cs",
            variants=("resnet18", "resnet34", "resnet50", "resnet101"),
            description="L2CS-Net gaze estimator (expects face boxes; no detector).",
        ),
        builder=build_l2cs_estimator,
    )
}


def available_estimators() -> List[str]:
    return sorted(_REGISTRY.keys())


def estimator_spec(name: str) -> EstimatorSpec:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown estimator: {name!r}. Available: {available_estimators()}")
    return _REGISTRY[key].spec


def available_variants(name: str) -> List[str]:
    return list(estimator_spec(name).variants)


def _normalize_device(device: str) -> str:
    """Resolve device strings.

    Supports:
      - 'auto' -> cuda:0 if available, else mps, else cpu
      - numeric strings like '0', '1' -> cuda:0, cuda:1 (if CUDA available)
      - passthrough values like 'cpu', 'mps', 'cuda', 'cuda:0'

    If a numeric device is provided but CUDA is unavailable, falls back to cpu.
    """
    raw = (device or "auto").strip()
    d = raw.lower()

    # Map numeric device to cuda:<idx>
    if d.isdigit():
        idx = int(d)
        try:
            if torch.cuda.is_available():
                return f"cuda:{idx}"
        except Exception:
            pass
        return "cpu"

    # Map plain 'cuda' to 'cuda:0'
    if d == "cuda":
        try:
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    if d != "auto":
        return raw

    # auto: cuda:0 if available, else mps, else cpu
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def _l2cs_arch_from_variant(variant: Optional[str]) -> str:
    """Map registry variant names to the L2CS Pipeline `arch` strings."""
    v = (variant or "").strip().lower()
    if not v:
        return "ResNet50"
    mapping = {
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "resnet101": "ResNet101",
    }
    return mapping.get(v, "ResNet50")


def get_gaze_estimator(
    *,
    estimator: str = "l2cs",
    variant: Optional[str] = None,
    weights: Path | str,
    device: str = "auto",
    expand_face: float = 0.0,
    **kwargs: Any,
) -> GazeEstimator:
    """Factory: construct a gaze estimator backend.

    Args:
      estimator: backend name (default: l2cs)
      variant: named variant for the backend (for l2cs, selects the backbone)
      weights: path to model weights
      device: 'auto' | 'cpu' | 'mps' | 'cuda' | 'cuda:0' | etc
      expand_face: expand each face box by this fraction before cropping

    Extra kwargs are accepted for forward-compat (ignored by unknown builders).
    """
    key = (estimator or "").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown estimator '{estimator}'. Available: {available_estimators()}")

    # Validate variant if provided
    if variant:
        v = variant.strip().lower()
        if v not in available_variants(key):
            raise ValueError(f"Unknown variant '{variant}' for estimator '{key}'. Available: {available_variants(key)}")

    arch = None
    if key == "l2cs":
        arch = _l2cs_arch_from_variant(variant)

    resolved_device = _normalize_device(device)

    entry = _REGISTRY[key]

    builder_kwargs: Dict[str, Any] = {
        "weights": Path(weights),
        "device": resolved_device,
        "expand_face": expand_face,
    }
    if key == "l2cs":
        builder_kwargs["arch"] = arch

    # Extra kwargs are accepted for forward-compat.
    builder_kwargs.update(kwargs)

    return entry.builder(**builder_kwargs)