# Mocopi utilities and scripts

This guide summarizes the current Mocopi helpers and the two dataset layouts they support.

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
sample_data/<sample_file>/
  session_log.jsonl
  <sample_file>_mocopi.bin
  phone_192.168.50.162/<sample_file>.mp4
  phone_192.168.50.171/<sample_file>.mp4
```

## Shared modules
- `mocopi.recording_io`: `load_mocopi_recording`, `load_mocopi_bin`, `resolve_mocopi_source`
- `mocopi.nd_pilot`: `CameraRecording`, `CaptureSession`, `TrialPair`, `discover_sessions`, `discover_pairs`, `resolve_session_pair`, `parse_camera_role_specs`
- `mocopi.sync`: `clean_feature_samples`, `estimate_camera_to_mocopi_offset`
- `mocopi.reliability`: `export_reliability_errors`, `ensure_reliability_csv`, `best_joint_from_reliability`, `nd_error_summary`, `get_aligned_traces`
- `mocopi.visualization`: `prepare_camera_landmarks`, `draw_camera_skeleton`, `prepare_mocopi_positions`, `draw_mocopi_skeleton`

## CLI conventions

### Motion input
- Scripts that previously required `--bvh` now accept `--motion`.
- `--motion` accepts:
  - a `.bvh` file
  - a session `.bin` file
  - a session directory containing `*_mocopi.bin`
- `--bvh` is still accepted as a compatibility alias in the direct motion-loading scripts.

### Session camera-role mapping
- Pair-based scripts need `--camera-role` when working with session folders.
- Use one argument per camera:
  - `--camera-role 192.168.50.162=A`
  - `--camera-role 192.168.50.171=ND`
- Accepted roles are `A` and `ND`.
- Mapping keys can be the bare camera id or the `phone_<camera_id>` directory name.

## CLI examples

- `scripts/mocopi_reliability_export.py`
  ```bash
  python -m scripts.mocopi_reliability_export \
    --motion sample_data/ND_pilot/'Re_ Mocopi'/MCPM_20251112_135620_1a.bvh \
    --camera_csv results/OutputCSVs/landmarks_ND_1a_20140107_104046.csv \
    --output results/mocopi_reliability/mocopi_camera_reliability_ND_1a_20140107_104046.csv
  ```

- `scripts/mocopi_sync_example.py`
  ```bash
  python -m scripts.mocopi_sync_example \
    --motion sample_data/<sample_file> \
    --camera_csv results/OutputCSVs/landmarks_<sample_file>.csv
  ```

- `scripts/mocopi_reliability_batch.py`
  ```bash
  python -m scripts.mocopi_reliability_batch \
    --camera-role 192.168.50.162=A \
    --camera-role 192.168.50.171=ND
  ```

- `scripts/mocopi_triplet_video.py`
  ```bash
  python -m scripts.mocopi_triplet_video \
    --tags <sample_file> \
    --camera-role 192.168.50.162=A \
    --camera-role 192.168.50.171=ND \
    --output-dir results/OutputVideos/triplets
  ```

- `scripts/mocopi_pair_report.py`
  ```bash
  python -m scripts.mocopi_pair_report \
    --camera-role 192.168.50.162=A \
    --camera-role 192.168.50.171=ND
  ```

- `scripts/mocopi_fourpanel_triplet.py`
  ```bash
  python -m scripts.mocopi_fourpanel_triplet \
    --tag <sample_file> \
    --camera-role 192.168.50.162=A \
    --camera-role 192.168.50.171=ND
  ```

- `scripts/mocopi_symmetry_diagnostics.py`
  ```bash
  python -m scripts.mocopi_symmetry_diagnostics \
    --camera-role 192.168.50.162=A \
    --camera-role 192.168.50.171=ND
  ```

## Outputs and conventions
- Camera CSVs are still expected under `results/OutputCSVs/landmarks_<video_stem>.csv`.
- Reliability CSVs are written under `results/mocopi_reliability/`.
- Video outputs are written under `results/OutputVideos/`.
- Legacy ND-factor parsing still comes from the ND video stem, so ND summary workflows remain meaningful only for legacy `ND_*` naming or any future session naming scheme that encodes ND level explicitly.
