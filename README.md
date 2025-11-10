# PatientPose - using mediapipe tools to develop patient movement profiles
## Instructions for installation
- If you have not already done so, install miniconda or a similar environment and package manager that comes with a git client installation
- In a terminal manager, navigate to the directory where you would like to make the project installation with a command such as
```bash
cd path\to\your\preferred\installation\location
```
- Once there, use git to clone the project to your preferred location, then cd into the project root:
```bash
git clone https://github.com/gerardjb/PatientPose
cd PatientPose
```
  * Next, make a dedicated environment for the project with calls similar to:
  ```bash
  conda create -n patient_pose python=3.12
  conda activate patient_pose
  ```
  * Note that as of this writing (20251103), mediapipe does not support pyhton > 3.12
  * Finally, use pip to install the project using either a system install:
  ```bash
  pip install .[test]
  ```
  * or a local install if you'd like to make edits to the codebase with:
  ```bash
  pip install -e .[test]
  ```
- In each of the pip installs, I've included the optional test suite which includes a pytest unit test - once the install is done, you can immedialtely use pytest to make sure everything is running properly as:
```bash
pytest
``` 
  * If you see green text with "4 passed", your installation is good to go. If not, you can send the error codes my way.
## Running the script
- I'm keeping project scripts in the "scripts" directory in the project root. So, to call any scripts from the command line, go there:
```bash
cd scripts
```
- Then call any scripts with commands like:
```bash
python the_script_you_want.py
```
  * For example, the sample_patient_processing.py will (barring future deprecation which will be a TODO here) create a de-identified version of the input video (removal of features near all detected faces) and a csv that captures the keypoints across the pose and hand landmark detector models.
  * All such files will be populated to the "results" subdirectory of the project root for further inspection if desired.
- For now, you can test a local file using the same flags as in the previous version, such as:
```bash
python sample_patient_processing.py --filename "your_file.extension"
```
  * If you'd like the auto-orientation stage to log every frame it inspected (including the per-rotation scores/landmarks), add the debug switches and point them to an output folder:
  ```bash
  python sample_patient_processing.py \
    --filename "/path/to/video.mp4" \
    --orientation-debug \
    --orientation-max-scan 250
  ```
    - Each run creates a JSON file in `results/orientation_debug/` (one per input) summarizing the selected rotation, the sampled frames, and the focus hint that seeds the ROI tracker.
  * To inspect the exact frames/rotations the pose-quality heuristics consider “good” or “bad”, use the helper script `debug_pose_quality_frames.py`:
  ```bash
  python debug_pose_quality_frames.py \
    --video sample_data/PXL_20251106_173128817_identifiable_and_rotated.mp4 \
    --pose-model models/pose_landmarker.task \
    --output-video results/pose_quality_debug.mp4
  ```
    - The CLI prints the per-rotation quality breakdown for the preset “bad” (0/10/20/30) and “good” (75/85/95) frames found in `sample_data\PXL_20251106_173128817_identifiable_and_rotated.mp4`, while the optional `results/pose_quality_debug.mp4` captures the annotated crops (frame number, rotation label, score, landmarks) for the test frames.
      - Note*: parameters are currently hard-coded within script itself, but I might update this as a TODO if this ends up being a persistent place driving issues.
- A labeling GUI for batching metadata entry lives at `scripts/video_labeling_gui.py`. It relies on the community-maintained `FreeSimpleGUI` package bundled with the project dependencies. Launch it from the `scripts` directory with:
```bash
python video_labeling_gui.py
```
  * On first launch, point the GUI at the folder containing the patient videos you want to label; the file list refreshes automatically when you change folders.
  * The form lets you enter the patient ID, pick a standardized exam type from the built-in recipes, adjust tags/notes, and apply the metadata to one or many selected videos at once. The latest values are stored in `results/labeling_state.json`, and `Undo` lets you revert the most recent save if needed.
  * Use `Process selected` to run the full de-identification pipeline on the highlighted clips. Outputs overwrite any prior runs and land in `results/OutputVideos/` and `results/OutputCSVs/`. Missing model files (e.g., `models/face_detection_yunet_2023mar.onnx` for the YuNet face detector that ships with the repo) or processing errors are reported inline.
  * When you're ready to export the current batch of labels, click `Export all` to write a timestamped JSON file in `results/label_exports/`.
  * Note: Mediapipe now attempts to load optional audio modules on import, which can hang if the system is missing PortAudio. All scripts set `MEDIAPIPE_SKIP_AUDIO=1` automatically, but if you run custom code be sure to export that environment variable (or install PortAudio) before launching Python.
## Updating the version of the repo on your local machine
Git allows you to update to the latest version of the codebase as long as you're in the project root as:
```bash
git pull
```
- As long as you haven't made any changes to the codebase in the version you have on your machine, this should update your version without errors
- If it doesn't work...see me, and we'll talk. :)
