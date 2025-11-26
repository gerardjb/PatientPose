# Mocopi utilities and scripts

This guide mirrors the main README style and summarizes the Mocopi helpers, their prerequisites, and the CLI patterns you’ll use most often.

## Prerequisites
- Camera CSVs (`results/OutputCSVs/landmarks_*.csv`) must already exist from your MediaPipe landmark extraction pipeline.
- Reliability/ND plots (`mocopi_reliability_export`, `mocopi_reliability_batch`, `mocopi_pair_report`, `mocopi_nd_summary_plot`, `mocopi_reliability_plot`) depend on reliability CSVs produced by the export/batch scripts.
- Video demos (`mocopi_side_by_side`, `mocopi_triplet_video`) need matching BVH + camera CSVs and either the raw ND/A MP4s in `sample_data/ND_pilot` or the pre-rendered AVIs noted below.

## Shared modules (highlights)
- `mocopi.nd_pilot`: `TrialPair`, `discover_pairs`, `pair_for_tag`
- `mocopi.sync`: `clean_feature_samples`, `estimate_camera_to_mocopi_offset`
- `mocopi.reliability`: `SCALE_REF_JOINTS`, `compute_body_scale_series`, `export_reliability_errors`, `ensure_reliability_csv`, `nd_factor_from_stem`, `best_joint_from_reliability`, `align_visibility_series`, `align_pose_counts`, `joint_medians`, `nd_error_summary`, `get_aligned_traces`
- `mocopi.camera_metrics`: `count_visible_landmarks`, `visibility_percent`
- `mocopi.visualization`: `prepare_camera_landmarks`, `draw_camera_skeleton`, `prepare_mocopi_positions`, `draw_mocopi_skeleton` (plus skeleton edge definitions)
- `mocopi.plots`: `select_overlap_window`, `plot_egocentric_compare`, `plot_feet_centered`

## CLI examples and expected outputs

- `scripts/mocopi_reliability_export.py`  
  ```bash
  python -m scripts.mocopi_reliability_export \
    --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
    --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \
    --output results/mocopi_reliability/mocopi_camera_reliability_ND_1a_20140107_104046.csv
  ```
  - Output: per-frame error CSV at the given path (egocentric, scale-normalized).

- `scripts/mocopi_reliability_batch.py`  
  ```bash
  python -m scripts.mocopi_reliability_batch --tags 1a 1b
  ```
  - Output: reliability CSVs under `results/mocopi_reliability/` using timestamped names (e.g., `mocopi_camera_reliability_ND_1a_20140107_104046.csv`).

- `scripts/mocopi_reliability_plot.py`  
  ```bash
  python -m scripts.mocopi_reliability_plot \
    --csv results/mocopi_reliability/mocopi_camera_reliability_ND_1a_20140107_104046.csv \
    --output results/mocopi_reliability_plot.pdf
  ```
  - Output: PDF bar plot of per-joint median error.

- `scripts/mocopi_nd_summary_plot.py`  
  ```bash
  python -m scripts.mocopi_nd_summary_plot \
    --inputs ND=2:results/mocopi_reliability/mocopi_camera_reliability_ND_1a_20140107_104046.csv \
             ND=4:results/mocopi_reliability/mocopi_camera_reliability_ND_1b_20140107_104202.csv \
    --output results/mocopi_nd_summary.pdf
  ```
  - Output: PDF line plot of median error vs ND (log2 x-axis), using the current timestamped reliability CSVs.

- `scripts/mocopi_pair_report.py`  
  ```bash
  python -m scripts.mocopi_pair_report --tags 1a --search_ms 5000 --rate_hz 50
  ```
  - Outputs: per-tag plots under `results/mocopi_reliability/plots/pair_<tag>_<joint>.pdf`, plus `results/mocopi_reliability/nd_delta_summary.csv` and `nd_ratio_summary.pdf`.

- `scripts/mocopi_fourpanel_triplet.py`  
  ```bash
  python -m scripts.mocopi_fourpanel_triplet --tag 1a --offset-ms 0 --output results/fourpanel_1a.pdf
  ```
  - Output: four-panel PDF (`results/fourpanel_1a_offset_0.0ms_vis_0.80.pdf`) showing Mocopi vs A vs ND egocentric ΔY and visibility percent.

- `scripts/mocopi_egocentric_compare.py`  
  ```bash
  python -m scripts.mocopi_egocentric_compare \
    --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
    --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \
    --output results/mocopi_camera_egocentric_ND_1a.png
  ```
  - Output: PNG with stacked Mocopi vs camera egocentric ΔY traces over aligned time.

- `scripts/mocopi_side_by_side.py`  
  ```bash
  python -m scripts.mocopi_side_by_side \
    --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
    --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \
    --video results/OutputVideos/deidentified_ND_1a_20140107_104046.avi \
    --output results/OutputVideos/mocopi_vs_camera_ND_1a.avi
  ```
  - Input note: uses the already-processed AVI in `results/OutputVideos` (orientation + overlay) instead of the raw `sample_data/ND_pilot/ND_1a_20140107_104046.mp4`.
  - Output: AVI with the provided processed camera panel on the left and Mocopi skeleton on the right.

- `scripts/mocopi_triplet_video.py`  
  ```bash
  python -m scripts.mocopi_triplet_video --tags 1a --output-dir results/OutputVideos/triplets
  ```
  - Output: AVI per tag (`results/OutputVideos/triplets/triplet_ND_1a_*.avi`) with A video + ND video + Mocopi panel (expects camera CSVs to exist).

- `scripts/mocopi_symmetry_diagnostics.py`  
  ```bash
  python -m scripts.mocopi_symmetry_diagnostics --tags 1a --search_ms 5000 --rate_hz 50
  ```
  - Output: correlation table to stdout and feet comparison PDFs under `results/mocopi_reliability/symmetry/feet_<tag>_<cond>.pdf`.

- `scripts/mocopi_sync_example.py`  
  ```bash
  python -m scripts.mocopi_sync_example \
    --bvh sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
    --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv
  ```
  - Output: console print of estimated camera→mocopi offset and correlation score.

- `scripts/mocopi_pair_utils.py`
  - Library-only shim; import via `from mocopi.nd_pilot import discover_pairs`.
