# Mocopi workflows

This document describes the current Mocopi-facing workflow surface in this repo.

The primary interface is now the packaged `patientpose` CLI:

```bash
patientpose --help
```

From the repo root, install the console entry point with:

```bash
pip install -e .
```

If the console script is not installed into the active environment yet, use:

```bash
python -m patientpose --help
```

Most of the old `scripts/*.py` entry points still exist as thin wrappers, but they are no longer the preferred surface.

## Supported data layouts

### 1. Legacy ND-pilot layout
- Videos live under `sample_data/ND_pilot/` as `A_<tag>_*.mp4` and `ND_<tag>_*.mp4`.
- Mocopi motion lives under `sample_data/ND_pilot/Re_ Mocopi/` as `MCPM_*_<tag>.bvh`.
- Pair discovery is automatic from the filename tag.

### 2. Session layout
- Each capture lives under `sample_data/<session_id>/`.
- Mocopi motion is stored as `*_mocopi.bin`.
- Camera videos are stored per phone under `phone_<camera_id>/VID_*.mp4`.
- Session metadata is stored in `session_log.jsonl`.
- Pair discovery requires explicit camera-role mapping because the video filenames do not encode `A` vs `ND`.

Example session:

```text
sample_data/<session_id>/
  session_log.jsonl
  <session_id>_mocopi.bin
  phone_192.168.50.162/VID_*.mp4
  phone_192.168.50.171/VID_*.mp4
```

## Camera role mapping

Session-mode pair discovery needs one `--camera-role` per phone:

```bash
--camera-role 192.168.50.162=A
--camera-role 192.168.50.171=ND
```

Accepted roles are `A` and `ND`. Mapping keys can be either the bare camera id or the `phone_<camera_id>` directory name.

## Preprocess artifacts

Run standard video preprocessing with:

```bash
patientpose preprocess video -f sample_data/test-1.MOV
```

Standard preprocessing writes:
- `results/OutputVideos/deidentified_<stem>.avi`
- `results/OutputVideos/deidentified_no_keypoints_<stem>.avi`
- `results/OutputCSVs/landmarks_<stem>.csv`
- `results/OutputCSVs/pose_world_<stem>.csv`
- `results/OutputCSVs/landmarks_summary_<stem>.csv`
- `results/OutputCSVs/landmarks_metadata_<stem>.json`

Notes:
- `landmarks_<stem>.csv` is the image-space landmark table used for overlays, QA, and video-linked tooling.
- `pose_world_<stem>.csv` is the world-space pose table used for gait-oriented diagnostics and Mocopi comparison.
- Both CSVs are written even when no pose landmarks are detected; the files will then be empty but schema-correct.
- `landmarks_metadata_<stem>.json` records orientation handling and linked artifacts.

The quality-video workflow remains available:

```bash
patientpose preprocess quality-video -f sample_data/20250408_fingerTap_decrement.mp4
```

That writes:
- `results/OutputVideos/quality_vis_<stem>.avi`
- `results/OutputVideos/quality_vis_no_keypoints_<stem>.avi`
- `results/OutputCSVs/landmarks_<stem>.csv`
- `results/OutputCSVs/pose_world_<stem>.csv`
- `results/OutputCSVs/landmarks_metadata_<stem>.json`
- `results/OutputPlots/fingertip_position_<stem>.png`
- `results/OutputPlots/fingertip_quality_<stem>.png`

## Analyze workflows

### Direct reliability export

Use this when you already know the exact motion file and camera CSV:

```bash
patientpose analyze reliability \
  --motion sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
  --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv
```

Useful options:
- `--camera-space image|world`
- `--world-csv <path>` if you want to override the paired `pose_world_*.csv`
- `--offset_ms` to force a known offset
- `--clip-start` / `--clip-end` to restrict offset estimation to a cleaner segment

### Batch reliability export

Use this to process discovered pairs:

```bash
patientpose analyze reliability-batch \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND
```

Useful options:
- `--tags 1a 2a` for legacy subsets
- `--tags <data root>` for session subsets
- `--camera-space image|world`
- `--output-dir results/mocopi_reliability`

## Report workflows

### Pair report

This generates per-pair plots and ND-vs-A summary outputs:

```bash
patientpose report pair-report \
  --tags <data root> \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND
```

Useful options:
- `--camera-space image|world`
- `--plot-component x|y|z`
- `--offset_ms` to force a shared offset
- `--clip_start` / `--clip_end` to restrict offset estimation and plots

Outputs live under `results/mocopi_reliability/`, including:
- `nd_delta_summary_<space>.csv`
- `nd_ratio_summary_<space>.pdf`
- `plots/<space>/pair_<tag>_<joint>.pdf`

## Render workflows

### Side-by-side video

Use this for a direct camera-vs-Mocopi panel render:

```bash
patientpose render side-by-side \
  --motion sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
  --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv
```

Useful options:
- `--video <path>` to override the inferred camera-panel video
- `--camera-panel-source auto|deidentified|deidentified-no-keypoints|raw`
- `--video-rotation auto|none|90cw|90ccw|180`
- `--mocopi-view body-centered|walk-range`

By default, `auto` camera panels prefer:
1. `deidentified_<stem>.avi`
2. `deidentified_no_keypoints_<stem>.avi`
3. the raw source video

### Triplet video

Use this for A / ND / Mocopi video triplets:

```bash
patientpose render triplet-video \
  --tags <data root> \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND
```

Useful options:
- `--camera-panel-source auto|deidentified|deidentified-no-keypoints|raw`
- `--offset_ms` to force a shared offset
- `--max_frames`
- `--video-rotation auto|none|90cw|90ccw|180`
- `--mocopi-view body-centered|walk-range`

Outputs default to:
- `results/OutputVideos/triplets/triplet_<tag>.avi`

### Four-panel triplet plot

Use this for A / ND / Mocopi egocentric trace comparison:

```bash
patientpose render fourpanel-triplet \
  --tag <data root> \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND
```

Useful options:
- `--camera-space image|world`
- `--plot-component x|y|z`
- `--camera-display-feature auto|raw|lower-limb-composite|distal-foot-composite|weighted-lower-limb`
- `--camera-display-smooth-window N`
- `--offset-ms <ms>` to override automatic sync

Current behavior:
- If `--offset-ms` is omitted, offsets are estimated by cross-correlation.
- The sync feature and the plotted camera feature are now separate concerns.
- Cross-correlation uses a scored candidate bank rather than a hard-coded wrist-only feature.
- Four-panel filenames encode camera space, plotted component, offset mode, and visibility threshold.

Example world-space gait view:

```bash
patientpose render fourpanel-triplet \
  --tag <data root> \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND \
  --camera-space world \
  --plot-component z \
  --camera-display-feature distal-foot-composite \
  --camera-display-smooth-window 9
```

Outputs default to:
- `results/OutputPlots/fourpanel_<tag>_<space>_d<component>_<offset-label>_vis_<threshold>.pdf`

## Diagnostics workflows

### Egocentric component plots

Use this to inspect camera projection quality directly:

```bash
patientpose diagnose egocentric-plot \
  --tag <data root> \
  --camera-side ND \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND \
  --space world \
  --components y z \
  --body-frame
```

Useful options:
- `--camera_csv <path>` instead of `--tag`
- `--space image|world`
- `--world_csv <path>`
- `--landmarks LEFT_ANKLE RIGHT_ANKLE`
- `--components x y z`
- `--smooth-window N`

### Egocentric overlay video

Use this to inspect the projected signal against the video:

```bash
patientpose diagnose egocentric-video \
  --tag <data root> \
  --camera-side ND \
  --camera-role 192.168.50.162=A \
  --camera-role 192.168.50.171=ND \
  --space world \
  --components y z \
  --body-frame
```

Useful options:
- `--video <path>` to override the inferred panel video
- `--camera-panel-source auto|deidentified|deidentified-no-keypoints|raw`
- `--video-rotation auto|none|90cw|90ccw|180`
- `--max-frames N`

Diagnostics outputs default to:
- `results/Diagnostics/egocentric/<stem>_<space>_<frame-mode>_components.pdf`
- `results/Diagnostics/egocentric/<stem>_<space>_<frame-mode>_overlay.avi`

## Image space vs world space

Use image-space pose when the task needs to line up with pixels:
- overlay rendering
- deidentification
- orientation checks
- frame-level QA

Use world-space pose when the task is primarily kinematic:
- gait diagnostics
- Mocopi comparison
- egocentric trace inspection
- world-space four-panel plots

In practice:
- `--camera-space world` is the better default for gait-oriented comparisons.
- `--space image` remains useful for troubleshooting whether the raw 2D tracking itself is the problem.

## Sync behavior

Current sync estimation is no longer based on a single hard-coded wrist trace.

Instead:
- camera-to-Mocopi sync uses a scored bank of candidate features
- gait-oriented world-space lower-limb features are preferred when available
- direct camera-to-camera offset is also estimated and reported in the render workflows

The render and report outputs now separate:
- sync feature selection
- plotted camera trace selection

This matters for gait, where a feature that is robust for offset estimation is not always the cleanest feature to display.

## Legacy wrappers

These packaged workflows currently still have thin wrapper scripts under `scripts/`, but prefer the `patientpose` command:
- `scripts/sample_patient_processing.py`
- `scripts/process_video_for_blur.py`
- `scripts/mocopi_reliability_export.py`
- `scripts/mocopi_reliability_batch.py`
- `scripts/mocopi_pair_report.py`
- `scripts/mocopi_side_by_side.py`
- `scripts/mocopi_triplet_video.py`
- `scripts/mocopi_fourpanel_triplet.py`
