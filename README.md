

# gaze-lib

**Minimum Python:** `>=3.10`

**gaze-lib** is a modular **gaze estimation + JSON augmentation** toolkit that attaches gaze predictions to detections containing face boxes.

This is the **Gaze Augmentation stage** of the Vision Pipeline.

Estimators included:
- **l2cs**: L2CS-Net backend (face-box driven; no internal detector)

> By default, `gaze-lib` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Vision Pipeline

```
Original Video (.mp4)
        │
        ▼
  detect-lib
  (Detection Stage)
        │
        └── detections.json (det-v1)
                   │
                   ▼
                track-lib
           (Tracking + ReID)
                   │
                   └── tracked.json (track-v1)
                           │
                           ▼
                    detect-face-lib
                 (Face Augmentation)
                           │
                           └── faces.json (face-v1 meta)
                                   │
                                   ▼
                              gaze-lib
                         (Gaze Augmentation)
                                   │
                                   └── gaze.json (augmented; gaze-v1 meta)
```

Note: Each stage consumes the original video + the upstream JSON from the previous stage.

---

## What gaze-lib expects

`gaze-lib` **does not run a face detector**.

Input JSON must contain:
- `frames[*].detections[*]`
- Inside detections: `faces` with valid `bbox`

The parent schema may be:
- `face-v1`
- `det-v1`
- `track-v1`
- or unknown

As long as detections and face boxes exist, normalization will adapt.

---

## Output: augmented JSON (returned + optionally saved)

`gaze-lib` returns an **augmented JSON payload** in-memory that preserves the upstream schema and adds:

- `gaze_augment`: metadata about the estimator + association rules (versioned)
- `detections[*].gaze`: minimal gaze payload

### What gets attached to a detection

Each gaze entry is intentionally minimal:

- `yaw` (radians)
- `pitch` (radians)
- `gaze_vec`: `[x,y,z]` unit vector
- `face_ind`: which face entry was used
- `origin`: `[x,y]` pixel location (if available)
- `origin_source`: `"kpt"` or `"box"`

No redundant or derivable data is stored.

---

## Minimal schema example

```json
{
  "schema_version": "track-v1",
  "gaze_augment": {
    "version": "gaze-v1",
    "parent_schema_version": "track-v1",
    "estimator": {
      "name": "l2cs",
      "variant": "resnet50",
      "weights": "weights.pkl",
      "device": "auto"
    }
  },
  "frames": [
    {
      "frame_index": 0,
      "detections": [
        {
          "bbox": [100.0, 50.0, 320.0, 240.0],
          "faces": [
            {
              "bbox": [140.0, 70.0, 210.0, 150.0],
              "score": 0.98
            }
          ],
          "gaze": {
            "yaw": -0.12,
            "pitch": 0.08,
            "gaze_vec": [0.11, -0.08, -0.99],
            "face_ind": 0
          }
        }
      ]
    }
  ]
}
```

---

## Returned vs saved

- **Returned (always):** payload available in memory via `GazeResult.payload`
- **Saved (opt-in):**
  - `--json` → `<run>/gaze.json`
  - `--frames` → `<run>/frames/`
  - `--save-video` → `<run>/annotated.mp4`

If no artifact flags are enabled, nothing is written.

---

# Install with pip (future PyPI)

Requires Python >= 3.10.

```bash
pip install gaze-lib
```

Module import name remains:

```python
import gaze
```

---

## L2CS Weights

Pretrained weights:
https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing

Currently supported variant:

- `resnet50`

If using custom weights, ensure they match the correct L2CS variant.

---

# CLI Usage (pip or installed package)

Global help:

```bash
python -m gaze.cli.estimate_gaze -h
```

List estimators:

```bash
python -m gaze.cli.estimate_gaze --list-estimators
```

List variants:

```bash
python -m gaze.cli.estimate_gaze --estimator l2cs --list-variants
```

---

## Quick Start

```bash
python -m gaze.cli.estimate_gaze \
  --json faces.json \
  --video in.mp4 \
  --weights weights.pkl
```

---


## Save artifacts (opt-in)

```bash
python -m gaze.cli.estimate_gaze \
  --json faces.json \
  --video in.mp4 \
  --weights weights.pkl \
  --json-out \
  --frames \
  --save-video annotated.mp4 \
  --out-dir out --run-name demo
```

---

## CLI arguments

### Required (for running augmentation)

- `--json-in <path>`: Path to the upstream JSON to augment.
  - Accepts `face-v1`, `det-v1`, `track-v1`, or unknown schemas as long as the JSON contains `frames[*].detections[*]`.
- `--video <path>`: Path to the original source video used to generate the upstream JSON. Frame order must align.
- `--weights <path>`: Path to L2CS weights (`.pkl`).

### Discovery

- `--list-estimators`: Print available gaze estimator backends and exit.
- `--list-variants`: Print supported variants for `--estimator` and exit.

### Estimator selection

- `--estimator <name>`: Gaze estimator backend to use (default: `l2cs`).
- `--variant <name>`: Backend variant (named variant registry).
  - For `l2cs`, this selects the backbone. **The pretrained weights linked above currently support only `resnet50`.**

### Face crop behavior

- `--expand-face <float>`: Expand each face box by this fraction before cropping.
  - Example: `--expand-face 0.25` expands width/height by +25%.
  - Increase → includes more context (forehead/hair/ears); can improve stability but may include background.
  - Decrease → tighter crop; can be sharper but may clip parts of the face.
  - Practical range: `0.0–0.35` (start around `0.2–0.3`).

### Association / filtering

- `--associate-classes <ids...>`: Only attach gaze to detections whose `class_id` is in this list.
  - Example: `--associate-classes 0` (often `person`).
  - If omitted, `gaze-lib` tries to infer `class_name == "person"`; if not found, all classes are eligible.
- `--face-index <int>`: Which face entry to use per detection.
  - If set, always uses that index when present.
  - If omitted, uses the **highest-score face** in `faces`.

### Gaze origin behavior (optional)

- `--kpt-origin <ids...>`: Keypoint indices (from `detections[*].keypoints`) used to compute a gaze origin.
  - Origin is computed as the **mean of the selected keypoints that pass confidence**.
  - Example: `--kpt-origin 0 1`.
- `--kpt-conf <float>`: Minimum keypoint confidence for origin computation (default: `0.3`).
  - Increase → fewer keypoints qualify (more robust, but more detections may fall back/skip).
  - Decrease → more keypoints qualify (more coverage, but noisier origins).
- `--fallback`: If set, when keypoint-origin is unavailable, fall back to the face box center (preferred) or detection box center.
  - If not set and `--kpt-origin` is provided, detections without a valid keypoint-origin are skipped.

If you pass `--kpt-origin` but the JSON contains no keypoints, `gaze-lib` emits a warning and continues.

### Artifact saving (all opt-in)

- `--json`: Write augmented JSON to `<run>/gaze.json`.
- `--frames`: Save annotated frames under `<run>/frames/`.
- `--save-video [name.mp4]`: Save an annotated video under `<run>/`.
- `--out-dir <dir>`: Output root used only when saving artifacts (default: `out`).
- `--run-name <name>`: Optional subfolder under `--out-dir`.
- `--fourcc <fourcc>`: FourCC codec for saved video (default: `mp4v`).
- `--display`: Show a live annotated preview (ESC to stop). Does not write files unless saving flags are set.

### UX

- `--no-progress`: Disable progress bar.

---

# Python usage (import)

You can use `gaze-lib` as a library after installing it.

### Quick sanity check

```bash
python -c "import gaze; print(gaze.available_estimators()); print(gaze.available_variants('l2cs'))"
```

### Python API reference (keywords)

#### `gaze.estimate_gaze_video(...)`

**Required**
- `json_in`: Path to the upstream JSON.
- `video`: Path to the original source video.
- `weights`: Path to L2CS weights (`.pkl`).

**Estimator**
- `estimator`: Backend name (default: `"l2cs"`).
- `variant`: Named variant for the backend.
  - For L2CS pretrained weights linked above, use `"resnet50"`.
- `device`: `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, `"cuda:0"`, or an index string like `"0"`.
- `expand_face`: Expand face crop by fraction (`0.0–0.35`, start `0.2–0.3`).

**Association / selection**
- `associate_class_ids`: List of `class_id` values eligible for gaze attachment.
  - If `None`, the tool tries to infer `class_name == "person"`; if not found, all classes are eligible.
- `face_index`: If set, use that face index per detection; otherwise choose the highest-score face.

**Origin (optional)**
- `kpt_origin`: List of keypoint indices used to compute gaze origin.
- `kpt_conf`: Keypoint confidence threshold.
- `fallback`: If `True`, fall back to box center when keypoint-origin is unavailable.

**Artifacts (all off by default)**
- `save_json_flag`: Write `<run>/gaze.json`.
- `save_frames`: Write `<run>/frames/*.jpg`.
- `save_video`: Filename for annotated video under the run folder.
- `out_dir`, `run_name`, `fourcc`, `display`, `no_progress`.

Returns a `GazeResult` with `payload` (augmented JSON), `paths` (only populated when saving), and `stats`.

### Run gaze augmentation from a Python file

Create `run_gaze.py`:

```python
from gaze import estimate_gaze_video

res = estimate_gaze_video(
    json_in="faces.json",
    video="in.mp4",
    estimator="l2cs",
    variant="resnet50",
    weights="weights.pkl",
    device="auto",

    # Optional filtering
    associate_class_ids=[0],

    # Optional crop tuning
    expand_face=0.25,

    # Optional origin behavior
    kpt_origin=[0, 1],
    kpt_conf=0.3,
    fallback=True,

    # Opt-in artifacts
    save_json_flag=True,
    save_video="annotated.mp4",
    out_dir="out",
    run_name="demo",
)

print(res.stats)
print("gaze_augment" in res.payload)
print(res.paths)  # populated only if you enable saving artifacts
```

Run:

```bash
python run_gaze.py
```

# Using uv (recommended for development)

Install uv:
https://docs.astral.sh/uv/

Clone the repo:
```bash
git clone https://github.com/Surya-Rayala/VisionPipeline-gaze.git
cd VisionPipeline-gaze
```

Sync environment:

```bash
uv sync
```

Run CLI:

```bash
uv run python -m gaze.cli.estimate_gaze -h
```

Run augmentation:

```bash
uv run python -m gaze.cli.estimate_gaze \
  --json faces.json \
  --video in.mp4 \
  --weights weights.pkl
```

---

# Testing different help levels

Basic help:

```bash
uv run python -m gaze.cli.estimate_gaze -h
```

Estimator help:

```bash
uv run python -m gaze.cli.estimate_gaze --list-estimators
```

Variant help:

```bash
uv run python -m gaze.cli.estimate_gaze --estimator l2cs --list-variants
```

---

# License

This project is licensed under the **MIT License**. See `LICENSE`.