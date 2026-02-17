from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GazeResult:
    """Result returned by gaze augmentation.

    - payload: augmented JSON (always in-memory)
    - paths: populated only when saving is enabled
    - stats: small counters / run info
    """

    payload: Dict[str, Any]
    paths: Dict[str, Path] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


def _maybe_path(p: Optional[str | Path]) -> Optional[Path]:
    if p is None:
        return None
    return p if isinstance(p, Path) else Path(p)