"""
Interactive labeling GUI for batching patient video annotations.

The tool is intentionally self-contained so non-technical teammates can
launch it, pick a directory of videos, and attach structured metadata that
is persisted to JSON. FreeSimpleGUI keeps the footprint small while providing
enough flexibility for future polish (packaging, previews, etc.).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import cv2

try:
    import FreeSimpleGUI as sg
except ImportError as exc:  # pragma: no cover - runtime convenience
    raise SystemExit(
        "FreeSimpleGUI is required for the labeling GUI. "
        "Install it with `pip install FreeSimpleGUI` and relaunch."
    ) from exc

try:
    from sample_patient_processing import (
        CSV_DIR,
        DEFAULT_MODEL_DIR,
        HAND_MODEL_FILENAME,
        POSE_MODEL_FILENAME,
        VIDEO_DIR,
        determine_rotation_code,
        process_video,
    )
except ImportError as exc:
    raise SystemExit(
        "Unable to import video processing utilities. "
        "Run this script from the repository's `scripts/` directory "
        "after installing project dependencies."
    ) from exc

# --- Paths & constants -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
EXPORT_DIR = RESULTS_DIR / "label_exports"
STATE_PATH = RESULTS_DIR / "labeling_state.json"
CONFIG_DIR = BASE_DIR / "config"
RECIPES_PATH = CONFIG_DIR / "labeling_recipes.json"
DEFAULT_VIDEO_ROOT = BASE_DIR / "sample_data"

# Video extensions we consider during folder scans
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Event keys
PROCESS_DONE_EVENT = "-PROCESS_DONE-"
PROCESS_STATUS_EVENT = "-PROCESS_STATUS-"

# Fallback recipe definitions; written out the first time if no config exists
DEFAULT_RECIPES: Dict[str, Dict[str, Any]] = {
    "gait_assessment": {
        "label": "Gait Assessment",
        "description": "Patient walking to assess gait and bradykinesia.",
        "tags": ["gait", "bradykinesia", "mobility"],
        "checklist": [
            "Include whether assistive devices are present.",
            "Note observed freezing episodes or hesitation.",
            "Capture stride length / symmetry observations.",
        ],
    },
    "finger_tapping": {
        "label": "Finger Tapping",
        "description": "Finger tapping task focusing on decrement and hesitations.",
        "tags": ["bradykinesia", "upper_extremity"],
        "checklist": [
            "Left, right, or bilateral task noted.",
            "Decrement severity scored or described.",
            "Any dyskinesias or tremor co-occurring.",
        ],
    },
    "rest_tremor": {
        "label": "Rest Tremor",
        "description": "Assessment of rest tremor at rest and with distraction.",
        "tags": ["tremor", "upper_extremity"],
        "checklist": [
            "Specify body side(s) involved.",
            "Describe amplitude and constancy.",
            "Mention distraction maneuver results.",
        ],
    },
}


# --- Data containers ---------------------------------------------------------


@dataclass
class LabelRecord:
    """Structured metadata captured for a single video."""

    video_path: str
    patient_id: str
    exam_type: str
    description: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    labeled_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelRecord":
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [segment.strip() for segment in tags.split(",") if segment.strip()]
        return cls(
            video_path=data["video_path"],
            patient_id=data.get("patient_id", ""),
            exam_type=data.get("exam_type", ""),
            description=data.get("description", ""),
            tags=list(tags),
            notes=data.get("notes", ""),
            labeled_at=data.get(
                "labeled_at", datetime.utcnow().isoformat(timespec="seconds")
            ),
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class LabelingSession:
    """Registry for all labeling progress and persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: Dict[str, LabelRecord] = {}
        self.history: List[Dict[str, Any]] = []
        self.settings: Dict[str, Any] = {}
        self._load()

    # --- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            sg.popup_error(
                "Unable to load existing labeling state.",
                f"{exc}",
                title="Labeling GUI",
            )
            return

        for raw_label in payload.get("records", []):
            record = LabelRecord.from_dict(raw_label)
            self.records[record.video_path] = record
        self.history = payload.get("history", [])
        self.settings = payload.get("settings", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "records": [record.to_json() for record in self.records.values()],
            "history": self.history[-50:],  # lightweight undo buffer
            "settings": self.settings,
        }
        self.path.write_text(json.dumps(json_payload, indent=2))

    # --- mutations -----------------------------------------------------------

    def set_label(self, record: LabelRecord) -> None:
        previous = self.records.get(record.video_path)
        self.records[record.video_path] = record
        self.history.append(
            {
                "video_path": record.video_path,
                "previous": previous.to_json() if previous else None,
            }
        )
        self.save()

    def clear_label(self, video_path: str) -> None:
        if video_path in self.records:
            previous = self.records.pop(video_path)
            self.history.append({"video_path": video_path, "previous": previous.to_json()})
            self.save()

    def undo(self) -> Optional[str]:
        if not self.history:
            return None
        last = self.history.pop()
        video_path = last["video_path"]
        previous = last["previous"]
        if previous:
            restored = LabelRecord.from_dict(previous)
            self.records[video_path] = restored
        else:
            self.records.pop(video_path, None)
        self.save()
        return video_path

    # --- queries -------------------------------------------------------------

    def get(self, video_path: str) -> Optional[LabelRecord]:
        return self.records.get(video_path)

    def is_labeled(self, video_path: str) -> bool:
        return video_path in self.records

    def labeled_count(self, video_paths: Iterable[str]) -> int:
        return sum(1 for path in video_paths if path in self.records)

    # --- exports -------------------------------------------------------------

    def export_batch(self, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_path = destination_dir / f"labels_{timestamp}.json"
        export_path.write_text(
            json.dumps([record.to_json() for record in self.records.values()], indent=2)
        )
        return export_path


# --- Recipe utilities --------------------------------------------------------


def ensure_recipe_file() -> Dict[str, Dict[str, Any]]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if RECIPES_PATH.exists():
        try:
            return json.loads(RECIPES_PATH.read_text())
        except json.JSONDecodeError:
            sg.popup_error(
                "Recipe file is not valid JSON; falling back to defaults.",
                title="Labeling GUI",
            )
    RECIPES_PATH.write_text(json.dumps(DEFAULT_RECIPES, indent=2))
    return DEFAULT_RECIPES


def format_recipe_hint(recipe: Dict[str, Any]) -> str:
    lines = [recipe.get("description", "")]
    checklist = recipe.get("checklist") or []
    if checklist:
        lines.append("")
        for item in checklist:
            lines.append(f"• {item}")
    return "\n".join(lines).strip()


# --- GUI utilities -----------------------------------------------------------


def discover_videos(root: Path) -> List[Path]:
    if not root.exists():
        return []
    paths = [
        path
        for path in root.rglob("*")
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()
    ]
    return sorted(paths)


def build_layout(recipes: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    recipe_labels = [recipes[key]["label"] for key in recipes]

    left_column = [
        [
            sg.Text("Video folder", size=(12, 1)),
            sg.Input(str(DEFAULT_VIDEO_ROOT), key="-FOLDER-", enable_events=True),
            sg.FolderBrowse(target="-FOLDER-"),
        ],
        [
            sg.Listbox(
                values=[],
                select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
                size=(45, 20),
                key="-VIDEO_LIST-",
                enable_events=True,
            )
        ],
    ]

    right_column = [
        [
            sg.Text("Patient ID"),
            sg.Input(key="-PATIENT_ID-", size=(25, 1)),
            sg.Button("Copy from last", key="-COPY_LAST-", tooltip="Reuse prior patient ID"),
        ],
        [
            sg.Text("Exam type"),
            sg.Combo(
                recipe_labels,
                readonly=True,
                key="-EXAM_TYPE-",
                enable_events=True,
                size=(30, 1),
            ),
        ],
        [
            sg.Text("Tags (comma separated)"),
            sg.Input(key="-TAGS-", size=(40, 1)),
        ],
        [
            sg.Text("Description", pad=(0, 8)),
        ],
        [
            sg.Multiline(
                key="-DESCRIPTION-",
                size=(45, 5),
                autoscroll=True,
            )
        ],
        [
            sg.Text("Checklist", pad=(0, 8)),
        ],
        [
            sg.Multiline(
                key="-RECIPE_HINT-",
                size=(45, 6),
                disabled=True,
                autoscroll=True,
            )
        ],
        [
            sg.Text("Notes"),
        ],
        [
            sg.Multiline(
                key="-NOTES-",
                size=(45, 4),
                autoscroll=True,
            )
        ],
    ]

    bottom_row = [
        sg.Button("Save label(s)", key="-SAVE-", bind_return_key=True),
        sg.Button("Process selected", key="-PROCESS-"),
        sg.Button("Undo last", key="-UNDO-"),
        sg.Button("Export all", key="-EXPORT-"),
        sg.Text("Progress: 0 / 0", key="-PROGRESS_LABEL-", pad=((20, 5), (0, 0))),
        sg.ProgressBar(
            max_value=100,
            orientation="h",
            size=(25, 20),
            key="-PROGRESS-",
        ),
        sg.Push(),
        sg.Button("Clear label", key="-CLEAR-"),
        sg.Button("Exit"),
    ]

    status_row = [
        sg.StatusBar("Ready", key="-STATUS-", relief=sg.RELIEF_SUNKEN, size=(80, 1)),
    ]

    layout = [
        [sg.Column(left_column), sg.VSeparator(), sg.Column(right_column)],
        [bottom_row],
        [status_row],
    ]
    return layout


# --- Event handlers ----------------------------------------------------------


def parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def set_status(window: sg.Window, message: str) -> None:
    window["-STATUS-"].update(message)


def update_progress(
    window: sg.Window, session: LabelingSession, video_paths: List[Path]
) -> None:
    total = len(video_paths)
    if total == 0:
        window["-PROGRESS_LABEL-"].update("Progress: 0 / 0")
        window["-PROGRESS-"].UpdateBar(0, max=100)
        return

    labeled = session.labeled_count(str(path) for path in video_paths)
    window["-PROGRESS_LABEL-"].update(f"Progress: {labeled} / {total}")
    percent = int((labeled / total) * 100)
    window["-PROGRESS-"].UpdateBar(percent)


def build_display_map(
    session: LabelingSession, video_paths: List[Path]
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in video_paths:
        status_icon = "✓" if session.is_labeled(str(path)) else "·"
        label = f"[{status_icon}] {path.name} — {path.parent.name}"
        mapping[label] = path
    return mapping


def populate_form(
    window: sg.Window, session: LabelingSession, video_path: Path, recipes: Dict[str, Dict[str, Any]]
) -> None:
    record = session.get(str(video_path))
    if record:
        window["-PATIENT_ID-"].update(value=record.patient_id)
        window["-EXAM_TYPE-"].update(value=recipes.get(record.exam_type, {}).get("label", record.exam_type))
        window["-TAGS-"].update(value=", ".join(record.tags))
        window["-DESCRIPTION-"].update(value=record.description)
        window["-NOTES-"].update(value=record.notes)
        recipe_key = record.exam_type
    else:
        window["-DESCRIPTION-"].update(value="")
        window["-NOTES-"].update(value="")
        window["-TAGS-"].update(value="")
        chosen = window["-EXAM_TYPE-"].get()
        recipe_key = resolve_recipe_key(recipes, chosen)

    apply_recipe_hint(window, recipes, recipe_key)


def resolve_recipe_key(recipes: Dict[str, Dict[str, Any]], label: str) -> Optional[str]:
    for key, recipe in recipes.items():
        if recipe.get("label") == label:
            return key
    return None


def apply_recipe_hint(
    window: sg.Window, recipes: Dict[str, Dict[str, Any]], key: Optional[str]
) -> None:
    if not key or key not in recipes:
        window["-RECIPE_HINT-"].update(value="")
        return
    recipe = recipes[key]
    window["-RECIPE_HINT-"].update(value=format_recipe_hint(recipe))


def ensure_models_present() -> Optional[str]:
    hand_model_path = DEFAULT_MODEL_DIR / HAND_MODEL_FILENAME
    pose_model_path = DEFAULT_MODEL_DIR / POSE_MODEL_FILENAME
    if not hand_model_path.is_file():
        return f"Hand model not found at {hand_model_path}"
    if not pose_model_path.is_file():
        return f"Pose model not found at {pose_model_path}"
    return None


FORWARD_PROGRESS_PATTERN = re.compile(r"Processed\s+(\d+)\s+frames?\.?", re.IGNORECASE)


def _describe_rotation(rotation_code: int | None) -> str:
    mapping = {
        None: "no rotation",
        cv2.ROTATE_90_CLOCKWISE: "90° clockwise",
        cv2.ROTATE_90_COUNTERCLOCKWISE: "90° counter-clockwise",
        cv2.ROTATE_180: "180°",
    }
    return mapping.get(rotation_code, "unknown rotation")


def _safe_frame_count(video_path: Path) -> Optional[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return frame_count if frame_count > 0 else None


class _ForwardPassLogger(io.TextIOBase):
    """Intercept stdout lines to surface frame progress updates."""

    def __init__(self, tee_stream: io.TextIOBase, line_handler: Callable[[str], None]) -> None:
        self._tee_stream = tee_stream
        self._line_handler = line_handler
        self._buffer = ""

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        if self._tee_stream:
            self._tee_stream.write(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._line_handler(line.rstrip())
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._tee_stream:
            self._tee_stream.flush()
        if self._buffer:
            self._line_handler(self._buffer.rstrip())
            self._buffer = ""


def _emit_status(callback: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
    if not callback:
        return
    try:
        callback(payload)
    except Exception:
        # Best effort; GUI may have closed or no longer accept events.
        pass


def process_videos_batch(
    video_paths: List[str], status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    hand_model_path = DEFAULT_MODEL_DIR / HAND_MODEL_FILENAME
    pose_model_path = DEFAULT_MODEL_DIR / POSE_MODEL_FILENAME

    results: List[Dict[str, Any]] = []
    success_count = 0
    total_jobs = len(video_paths)

    for idx, path_str in enumerate(video_paths, start=1):
        video_path = Path(path_str)
        total_frames = _safe_frame_count(video_path)
        orientation_complete = False
        base_payload = {
            "video": str(video_path),
            "position": idx,
            "total_videos": total_jobs,
        }
        try:
            _emit_status(
                status_callback,
                {**base_payload, "phase": "orientation", "state": "start"},
            )
            rotation_code, pose_focus_hint = determine_rotation_code(
                video_path,
                pose_model_path,
                rotate_flag=False,
                auto_orient=True,
                return_details=True,
            )
            orientation_complete = True
            _emit_status(
                status_callback,
                {
                    **base_payload,
                    "phase": "orientation",
                    "state": "complete",
                    "rotation_label": _describe_rotation(rotation_code),
                },
            )
            _emit_status(
                status_callback,
                {
                    **base_payload,
                    "phase": "forward-pass",
                    "state": "start",
                    "total_frames": total_frames,
                },
            )

            def handle_forward_line(line: str) -> None:
                match = FORWARD_PROGRESS_PATTERN.search(line)
                if not match:
                    return
                processed_frames = int(match.group(1))
                percent = None
                if total_frames:
                    percent = min(int((processed_frames / total_frames) * 100), 100)
                _emit_status(
                    status_callback,
                    {
                        **base_payload,
                        "phase": "forward-pass",
                        "state": "progress",
                        "frames": processed_frames,
                        "total_frames": total_frames,
                        "percent": percent,
                    },
                )

            forward_logger = _ForwardPassLogger(sys.stdout, handle_forward_line)
            with contextlib.redirect_stdout(forward_logger):
                process_video(
                    video_path,
                    hand_model_path,
                    pose_model_path,
                    rotation_code,
                    pose_focus_hint=pose_focus_hint,
                )
            _emit_status(
                status_callback,
                {
                    **base_payload,
                    "phase": "forward-pass",
                    "state": "complete",
                    "frames": total_frames,
                    "total_frames": total_frames,
                    "percent": 100 if total_frames else None,
                },
            )
            output_video = VIDEO_DIR / f"deidentified_{video_path.stem}.avi"
            output_csv = CSV_DIR / f"landmarks_{video_path.stem}.csv"
            success_count += 1
            results.append(
                {
                    "video": str(video_path),
                    "status": "success",
                    "output_video": str(output_video),
                    "output_csv": str(output_csv) if output_csv.exists() else None,
                }
            )
        except Exception as exc:  # pragma: no cover - rely on runtime feedback
            phase = "forward-pass" if orientation_complete else "orientation"
            _emit_status(
                status_callback,
                {
                    **base_payload,
                    "phase": phase,
                    "state": "error",
                    "error": str(exc),
                },
            )
            results.append(
                {
                    "video": str(video_path),
                    "status": "error",
                    "error": str(exc),
                }
            )

    return {"success": success_count, "total": len(video_paths), "results": results}


def handle_processing_status(window: sg.Window, payload: Optional[Dict[str, Any]]) -> None:
    if not payload:
        return
    video_path = payload.get("video")
    video_name = Path(video_path).name if video_path else "Processing"
    position = payload.get("position")
    total_videos = payload.get("total_videos")
    if isinstance(position, int) and isinstance(total_videos, int) and total_videos > 0:
        label = f"{video_name} ({position}/{total_videos})"
    else:
        label = video_name
    phase = payload.get("phase")
    state = payload.get("state")

    message = ""

    if phase == "orientation":
        if state == "start":
            message = f"{label}: Assessing orientation..."
        elif state == "complete":
            rotation_label = payload.get("rotation_label")
            suffix = f" ({rotation_label})" if rotation_label else ""
            message = f"{label}: Orientation assessment complete{suffix}."
        elif state == "error":
            message = f"{label}: Orientation failed — {payload.get('error', 'Unknown error')}."
    elif phase == "forward-pass":
        if state == "start":
            total_frames = payload.get("total_frames")
            if total_frames:
                message = f"{label}: Forward pass started ({total_frames} frames)."
            else:
                message = f"{label}: Forward pass started."
        elif state == "progress":
            percent = payload.get("percent")
            frames = payload.get("frames")
            total_frames = payload.get("total_frames")
            if percent is not None and frames is not None and total_frames:
                message = f"{label}: Forward pass {percent}% ({frames}/{total_frames} frames)."
            elif frames is not None:
                message = f"{label}: Forward pass processed {frames} frames."
            else:
                message = f"{label}: Forward pass in progress..."
        elif state == "complete":
            message = f"{label}: Forward pass complete."
        elif state == "error":
            message = f"{label}: Forward pass failed — {payload.get('error', 'Unknown error')}."

    if message:
        set_status(window, message)


def main() -> None:
    recipes = ensure_recipe_file()
    sg.theme("SystemDefaultForReal")

    session = LabelingSession(STATE_PATH)
    layout = build_layout(recipes)
    window = sg.Window("Patient Video Labeling", layout, finalize=True, resizable=True)

    # Keep latest patient ID for "Copy from last" helper
    last_patient_id = ""
    processing_active = False

    # Initial discovery based on default folder
    current_folder = Path(window["-FOLDER-"].get())
    video_paths = discover_videos(current_folder)
    display_map = build_display_map(session, video_paths)
    window["-VIDEO_LIST-"].update(values=list(display_map.keys()))
    update_progress(window, session, video_paths)

    while True:
        event, values = window.read(timeout=200)

        if event in (sg.WINDOW_CLOSED, "Exit"):
            set_status(window, "Goodbye!")
            break

        if event == PROCESS_STATUS_EVENT:
            handle_processing_status(window, values.get(PROCESS_STATUS_EVENT))
            continue

        if event == "-FOLDER-":
            current_folder = Path(values["-FOLDER-"]).expanduser()
            video_paths = discover_videos(current_folder)
            display_map = build_display_map(session, video_paths)
            window["-VIDEO_LIST-"].update(values=list(display_map.keys()))
            update_progress(window, session, video_paths)
            set_status(window, f"Found {len(video_paths)} videos.")
            continue

        if event == "-VIDEO_LIST-":
            selections = values["-VIDEO_LIST-"]
            if selections:
                first_selection = selections[0]
                video_path = display_map.get(first_selection)
                if video_path:
                    populate_form(window, session, video_path, recipes)
                    set_status(window, f"Loaded metadata for {video_path.name}.")
            continue

        if event == "-EXAM_TYPE-":
            label = values["-EXAM_TYPE-"]
            key = resolve_recipe_key(recipes, label)
            if key:
                recipe = recipes[key]
                apply_recipe_hint(window, recipes, key)
                if not values["-DESCRIPTION-"].strip():
                    window["-DESCRIPTION-"].update(value=recipe.get("description", ""))
                if not values["-TAGS-"].strip():
                    window["-TAGS-"].update(value=", ".join(recipe.get("tags", [])))
            continue

        if event == "-COPY_LAST-":
            window["-PATIENT_ID-"].update(value=last_patient_id)
            continue

        if event == "-SAVE-":
            selections = values["-VIDEO_LIST-"]
            if not selections:
                set_status(window, "Select at least one video to label.")
                continue

            patient_id = values["-PATIENT_ID-"].strip()
            if not patient_id:
                set_status(window, "Patient ID is required.")
                continue

            exam_label = values["-EXAM_TYPE-"]
            exam_key = resolve_recipe_key(recipes, exam_label) or exam_label

            tags = parse_tags(values["-TAGS-"])
            description = values["-DESCRIPTION-"].strip()
            notes = values["-NOTES-"].strip()

            applied = 0
            for display in selections:
                video_path = display_map.get(display)
                if not video_path:
                    continue
                record = LabelRecord(
                    video_path=str(video_path.resolve()),
                    patient_id=patient_id,
                    exam_type=exam_key,
                    description=description,
                    tags=tags,
                    notes=notes,
                )
                session.set_label(record)
                applied += 1

            last_patient_id = patient_id
            display_map = build_display_map(session, video_paths)
            window["-VIDEO_LIST-"].update(values=list(display_map.keys()))
            update_progress(window, session, video_paths)
            set_status(window, f"Saved labels for {applied} video(s). Auto-saved session.")
            continue

        if event == "-PROCESS-":
            if processing_active:
                set_status(window, "Processing already in progress; please wait.")
                continue

            selections = values["-VIDEO_LIST-"]
            if not selections:
                set_status(window, "Select at least one video to process.")
                continue

            job_paths = []
            for display in selections:
                video_path = display_map.get(display)
                if video_path:
                    job_paths.append(str(video_path.resolve()))

            if not job_paths:
                set_status(window, "Unable to resolve selected video paths.")
                continue

            model_error = ensure_models_present()
            if model_error:
                sg.popup_error("Model files missing", model_error, title="Processing")
                set_status(window, "Processing aborted due to missing models.")
                continue

            processing_active = True
            window["-PROCESS-"].update(disabled=True)
            set_status(window, f"Processing {len(job_paths)} video(s)...")

            def status_callback(payload: Dict[str, Any]) -> None:
                try:
                    window.write_event_value(PROCESS_STATUS_EVENT, payload)
                except Exception:
                    pass
            if hasattr(window, "perform_long_operation"):
                window.perform_long_operation(
                    lambda paths=job_paths, cb=status_callback: process_videos_batch(paths, status_callback=cb),
                    PROCESS_DONE_EVENT,
                )
            else:  # pragma: no cover - fallback path
                result = process_videos_batch(job_paths, status_callback=status_callback)
                window.write_event_value(PROCESS_DONE_EVENT, result)
            continue

        if event == "-CLEAR-":
            selections = values["-VIDEO_LIST-"]
            cleared = 0
            for display in selections:
                video_path = display_map.get(display)
                if not video_path:
                    continue
                session.clear_label(str(video_path.resolve()))
                cleared += 1
            display_map = build_display_map(session, video_paths)
            window["-VIDEO_LIST-"].update(values=list(display_map.keys()))
            update_progress(window, session, video_paths)
            set_status(window, f"Cleared labels for {cleared} video(s).")
            continue

        if event == "-UNDO-":
            restored = session.undo()
            if restored:
                display_map = build_display_map(session, video_paths)
                window["-VIDEO_LIST-"].update(values=list(display_map.keys()))
                update_progress(window, session, video_paths)
                set_status(window, f"Undo applied for {Path(restored).name}.")
            else:
                set_status(window, "Nothing to undo.")
            continue

        if event == "-EXPORT-":
            export_path = session.export_batch(EXPORT_DIR)
            set_status(window, f"Exported {len(session.records)} labels to {export_path.name}.")
            sg.popup("Labels exported", f"Saved to: {export_path}", title="Export complete")
            continue

        if event == PROCESS_DONE_EVENT:
            processing_active = False
            window["-PROCESS-"].update(disabled=False)
            result = values.get(PROCESS_DONE_EVENT) or {}
            total = int(result.get("total", 0))
            success = int(result.get("success", 0))
            failures = max(total - success, 0)

            status_message = f"Processed {success} of {total} video(s)."

            if failures:
                error_lines = []
                for item in result.get("results", []):
                    if item.get("status") == "error":
                        video_name = Path(item.get("video", "")).name
                        error_lines.append(f"{video_name}: {item.get('error', 'Unknown error')}")
                sg.popup_error(
                    "Processing completed with errors",
                    "\n".join(error_lines) if error_lines else "Unknown processing error.",
                    title="Processing summary",
                )
            else:
                success_lines = []
                for item in result.get("results", []):
                    if item.get("status") != "success":
                        continue
                    video_name = Path(item.get("video", "")).name
                    video_output = item.get("output_video") or "Video output unavailable"
                    csv_output = item.get("output_csv") or "CSV output unavailable"
                    success_lines.append(
                        f"{video_name}\n  Video: {video_output}\n  CSV: {csv_output}"
                    )
                sg.popup(
                    "Processing complete",
                    "\n\n".join(success_lines) if success_lines else "Videos processed.",
                    title="Processing summary",
                )

            set_status(window, status_message)
            continue

    window.close()


if __name__ == "__main__":
    main()
