# `mocopi_com_pearson.py`

Standalone trial script for comparing a selected Mocopi trace against a selected MediaPipe camera trace, estimating a camera-to-Mocopi time offset, and exporting aligned traces, summary metrics, and diagnostic plots.

This script lives in `scripts/` because it is currently a workflow experiment rather than a finalized repo-wide surface.

## What It Does

Given:

- one Mocopi motion source
- one camera landmarks CSV

the script can:

- define a Mocopi trace from COM, one joint, a midpoint, or a difference
- define a camera trace from the projection origin, one landmark, a midpoint, or a difference
- estimate `offset_ms` using weighted cross-correlation
- optionally detrend traces before correlation
- optionally use a separate sync signal from the report/display signal
- optionally restrict sync to front-facing epochs
- export:
  - aligned trace CSV
  - offset scan CSV
  - one-row summary CSV
  - PNG diagnostic plot

## Basic Call

```powershell
python -m scripts.mocopi_com_pearson `
  --motion <mocopi.bvh|mocopi.bin|session_dir> `
  --camera_csv <landmarks.csv>
```

If `--world_csv` is omitted, the script tries to infer the paired `pose_world_*.csv`.

## Core Inputs

### Mocopi Trace

Use:

- `--mocopi-trace com`
- `--mocopi-trace single --mocopi-joints <joint>`
- `--mocopi-trace midpoint --mocopi-joints <joint1> <joint2>`
- `--mocopi-trace difference --mocopi-joints <left> <right>`

Useful defaults for gait:

- `l_foot`
- `r_foot`

### Camera Trace

Use:

- `--camera-trace origin`
- `--camera-trace single --camera-landmarks <landmark>`
- `--camera-trace midpoint --camera-landmarks <landmark1> <landmark2>`
- `--camera-trace difference --camera-landmarks <left> <right>`
- `raw-single`, `raw-midpoint`, `raw-difference` for raw camera-space coordinates instead of projected/body-scale-relative coordinates

Useful gait landmarks:

- `LEFT_ANKLE`
- `RIGHT_ANKLE`

### Coordinate Space And Component

Use:

- `--camera-space image|world`
- `--component x|y|z`

Defaults:

- `z` for `world`
- `y` for `image`

You can also override separately with:

- `--mocopi-component`
- `--camera-component`

## Offset Estimation

If `--offset-ms` is not provided, the script estimates an offset by cross-correlation.

Important convention:

```text
t_camera_aligned = t_camera + offset_ms
```

Controls:

- `--search-ms`
- `--step-ms`
- `--rate-hz`
- `--correlation-mode absolute|positive|negative`

## Front-Facing Gating

The script computes a `front_facing_score` from body geometry in image space and can use it to weight or gate the offset search.

Controls:

- `--front-weight-mode soft|hard|none`
- `--front-facing-threshold`
- `--front-window START_S END_S`
- `--front-segment all|first`
- `--front-segment-trim-start-s`
- `--front-segment-trim-end-s`

Typical usage:

- `all`: use every above-threshold front-facing run
- `first`: use only the first connected above-threshold run
- trims: shave the beginning/end off the kept run to remove sit-to-stand or turnaround contamination

## Detrending

Use detrending when the trial contains slow baseline changes that are not the gait signal of interest.

Controls:

- `--detrend none|rolling-mean|rolling-median`
- `--detrend-window-s`

Notes:

- raw traces are still plotted
- detrended traces are used for offset estimation and Pearson calculation

## Separate Sync Signal

The script can estimate offset from one signal and report another.

Controls:

- `--sync-signal trace|left-right-difference`
- `--sync-mocopi-joints LEFT RIGHT`
- `--sync-camera-landmarks LEFT RIGHT`

This is useful for gait when:

- report trace: `l_foot` vs `LEFT_ANKLE`
- sync trace: `(l_foot - r_foot)` vs `(LEFT_ANKLE - RIGHT_ANKLE)`

## Common Patterns

### 1. COM vs camera origin

```powershell
python -m scripts.mocopi_com_pearson `
  --motion sample_data\...\trial_mocopi.bin `
  --camera_csv results\OutputCSVs\landmarks_<stem>.csv `
  --camera-space world `
  --component z `
  --mocopi-trace com `
  --camera-trace origin
```

### 2. Left foot vs left ankle in image-space Y

```powershell
python -m scripts.mocopi_com_pearson `
  --motion sample_data\...\trial_mocopi.bin `
  --camera_csv results\OutputCSVs\landmarks_<stem>.csv `
  --camera-space image `
  --component y `
  --mocopi-trace single `
  --mocopi-joints l_foot `
  --camera-trace single `
  --camera-landmarks LEFT_ANKLE
```

### 3. Gait-oriented sync with left-right difference and detrending

```powershell
python -m scripts.mocopi_com_pearson `
  --motion sample_data\...\trial_mocopi.bin `
  --camera_csv results\OutputCSVs\landmarks_<stem>.csv `
  --camera-space image `
  --component y `
  --mocopi-trace single `
  --mocopi-joints l_foot `
  --camera-trace single `
  --camera-landmarks LEFT_ANKLE `
  --sync-signal left-right-difference `
  --sync-mocopi-joints l_foot r_foot `
  --sync-camera-landmarks LEFT_ANKLE RIGHT_ANKLE `
  --detrend rolling-median `
  --detrend-window-s 1.5
```

### 4. First connected front-facing bout only, trimmed

```powershell
python -m scripts.mocopi_com_pearson `
  --motion sample_data\...\trial_mocopi.bin `
  --camera_csv results\OutputCSVs\landmarks_<stem>.csv `
  --camera-space image `
  --component y `
  --mocopi-trace single `
  --mocopi-joints l_foot `
  --camera-trace single `
  --camera-landmarks LEFT_ANKLE `
  --sync-signal left-right-difference `
  --sync-mocopi-joints l_foot r_foot `
  --sync-camera-landmarks LEFT_ANKLE RIGHT_ANKLE `
  --detrend rolling-median `
  --detrend-window-s 1.5 `
  --front-segment first `
  --front-segment-trim-start-s 0.75 `
  --front-segment-trim-end-s 0.50 `
  --correlation-mode positive
```

## Outputs

By default, outputs go to:

```text
results/mocopi_reliability/com_pearson/
```

Or to `--output-dir` if provided.

Generated files:

- `*_aligned_traces.csv`
- `*_summary.csv`
- `*_offset_scan.csv`
- `*_aligned_traces.png`

If the auto-generated filename would get too long on Windows, the script falls back to a shorter hashed label.

## CSV Contents

### Aligned Trace CSV

Typical columns:

- `time_s`
- `mocopi_trace`
- `camera_trace`
- `mocopi_eval_trace`
- `camera_eval_trace`
- `mocopi_zscore`
- `camera_zscore`
- `mocopi_eval_zscore`
- `camera_eval_zscore`
- `sync_mocopi_trace`
- `sync_camera_trace`
- `front_facing_score`
- `front_facing_used`
- `offset_ms`

Interpretation:

- `*_trace`: raw aligned trace used for display
- `*_eval_trace`: detrended aligned trace used for Pearson
- `sync_*`: signal used for offset estimation

### Summary CSV

Includes:

- input provenance
- trace definitions
- sync definition
- detrend settings
- front-score selection settings
- selected segment bounds
- selected offset
- Pearson outputs
- output file paths

### Offset Scan CSV

Includes:

- `offset_ms`
- `signed_correlation`
- `selection_metric`
- `n_effective`
- `coverage`

## Plot Contents

The PNG contains:

- full raw Mocopi and camera traces
- full `front_facing_score`
- the aligned window
- aligned eval traces used for Pearson
- front-facing mask
- offset scan curve when offset is estimated

## Practical Notes

- A high peak at the search boundary is usually a warning, not a trustworthy sync.
- Constant baseline offsets do not by themselves hurt Pearson; slow drift and task-state changes do.
- For gait, `left-right-difference` plus detrending is often more useful than comparing absolute foot height directly.
- `front-segment first` means first connected above-threshold front-score run, not necessarily first clean gait run. Use trims to remove transient contamination.
