"""
Local dashboard for the Scene-Synced Slideshow Video Builder.

Run with:
    python app.py

Opens a browser tab at http://127.0.0.1:7860 where you can upload the script,
the voiceover, and the scene images (either a folder path on this machine, or
a .zip of images named 1.png, 2.jpg, ... one per scene), then render and
download the finished video. Everything runs locally - no cloud calls.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path

import gradio as gr

from pipeline import (
    build_video,
    EFFECTS,
    EFFECT_LABELS,
    TRANSITIONS,
    TRANSITION_LABELS,
    MIN_DURATION_FOR_TRANSITION,
)

APP_TMP = Path(tempfile.gettempdir()) / "scene_video_builder"
APP_TMP.mkdir(exist_ok=True)

ALL_OPTION = "All"
EFFECT_CHOICES = [ALL_OPTION] + [EFFECT_LABELS[e] for e in EFFECTS]
TRANSITION_CHOICES = [ALL_OPTION] + [TRANSITION_LABELS[t] for t in TRANSITIONS]
_LABEL_TO_EFFECT = {v: k for k, v in EFFECT_LABELS.items()}
_LABEL_TO_TRANSITION = {v: k for k, v in TRANSITION_LABELS.items()}


def _resolve_selection(selected_labels, label_to_key, full_keys) -> list[str]:
    """Empty selection -> field disabled ([]). 'All' present -> every key."""
    if not selected_labels:
        return []
    if ALL_OPTION in selected_labels:
        return full_keys[:]
    return [label_to_key[label] for label in selected_labels if label in label_to_key]


def _make_all_toggle_handler(real_labels: list[str]):
    """Keeps an 'All' multiselect option in sync with its individual options:
    checking 'All' ticks every option; unticking any one option drops 'All';
    unticking 'All' itself clears the whole field."""

    def handler(new_value, prev_value):
        # Filter stray None entries defensively: Gradio's multiselect Dropdown can
        # occasionally hand back a list containing None during rapid add/remove
        # churn; letting one through would round-trip into an invalid component value.
        new_value = [v for v in (new_value or []) if v is not None]
        prev_value = [v for v in (prev_value or []) if v is not None]
        had_all = ALL_OPTION in prev_value
        has_all = ALL_OPTION in new_value

        if has_all and not had_all:
            result = [ALL_OPTION] + real_labels[:]
        elif had_all and not has_all:
            result = []
        elif had_all and has_all and (set(prev_value) - set(new_value) - {ALL_OPTION}):
            result = [v for v in new_value if v != ALL_OPTION]
        else:
            result = new_value

        # If the correction is already reflected (same set of selections, order
        # aside), skip re-emitting the value: writing it again would re-trigger this
        # same .change() handler and loop forever between "correct" and "re-fired".
        if set(result) == set(new_value):
            return gr.skip(), result
        return gr.update(value=result), result

    return handler


def _prepare_images_dir(images_zip, images_folder_path: str) -> str:
    if images_folder_path and Path(images_folder_path).is_dir():
        return images_folder_path
    if images_zip is not None:
        extract_dir = Path(tempfile.mkdtemp(dir=APP_TMP, prefix="images_"))
        with zipfile.ZipFile(images_zip.name) as zf:
            zf.extractall(extract_dir)
        # If the zip contained a single wrapper folder, descend into it.
        entries = list(extract_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return str(entries[0])
        return str(extract_dir)
    raise gr.Error("Provide either a folder path or a .zip of scene images.")


def _safe_filename(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip().strip(".")
    if not name:
        raise gr.Error("Please enter a video title.")
    return name


def _unique_path(path: Path) -> Path:
    """Avoid clobbering an existing file: title.mp4 -> title (1).mp4 -> title (2).mp4 ..."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def run_pipeline(
    script_file,
    audio_file,
    images_zip,
    images_folder_path,
    width,
    height,
    fps,
    model_size,
    effects_selected,
    transitions_selected,
    captions,
    video_title,
    output_folder,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        if script_file is None:
            raise gr.Error("Please upload a script (.txt) file.")
        if audio_file is None:
            raise gr.Error("Please upload the voiceover audio file.")
        if not output_folder or not output_folder.strip():
            raise gr.Error("Please choose an output folder.")

        images_dir = _prepare_images_dir(images_zip, images_folder_path)

        out_dir = Path(output_folder).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise gr.Error(f"Could not create/use output folder {out_dir}: {e}")
        out_path = str(_unique_path(out_dir / f"{_safe_filename(video_title)}.mp4"))

        # The user can pick any folder at request time, so it can't be declared via
        # launch(allowed_paths=...) up front. Gradio re-checks this list on every
        # request, so registering it here (once known) is enough to let gr.Video
        # serve a file living outside the app's cwd/temp dir.
        out_dir_str = str(out_dir)
        if out_dir_str not in demo.allowed_paths:
            demo.allowed_paths.append(out_dir_str)

        log_lines: list[str] = []

        def progress_cb(msg: str, frac: float | None = None):
            log_lines.append(msg)
            progress(frac if frac is not None else 0, desc=msg)

        effects = _resolve_selection(effects_selected, _LABEL_TO_EFFECT, EFFECTS)
        transitions = _resolve_selection(transitions_selected, _LABEL_TO_TRANSITION, TRANSITIONS)

        build_video(
            script_path=script_file.name,
            audio_path=audio_file.name,
            images_dir=images_dir,
            out_path=out_path,
            resolution=(int(width), int(height)),
            fps=int(fps),
            model_size=model_size,
            captions=captions,
            effects=effects,
            transitions=transitions,
            progress_cb=progress_cb,
        )

        return out_path, "\n".join(log_lines)
    except gr.Error:
        raise
    except Exception as e:  # surface full traceback in the log box for debugging
        return None, f"ERROR: {e}\n\n{traceback.format_exc()}"


with gr.Blocks(title="Scene-Synced Slideshow Video Builder") as demo:
    gr.Markdown(
        "# Scene-Synced Slideshow Video Builder\n"
        "Upload your script, your single continuous voiceover, and your scene-numbered "
        "images (`1.png`, `2.jpg`, ...). Each image will be shown for the exact duration "
        "its scene is spoken, with rotating pan/zoom effects and word-by-word captions "
        "burned in. Everything runs locally on this machine."
    )

    with gr.Row():
        with gr.Column():
            script_file = gr.File(label="Script (.txt)", file_types=[".txt"])
            audio_file = gr.File(label="Voiceover (.mp3/.wav/.m4a)")
            gr.Markdown("**Scene images** — provide ONE of the two options below:")
            images_folder_path = gr.Textbox(
                label="Images folder path (on this machine)",
                placeholder="/path/to/images  (fastest for 200-300 images)",
            )
            images_zip = gr.File(label="...or a .zip of scene images", file_types=[".zip"])

        with gr.Column():
            video_title = gr.Textbox(
                label="Video title",
                placeholder="my-awesome-video",
                info="Used as the output filename.",
            )
            output_folder = gr.Textbox(
                label="Output folder",
                value=str(Path.home() / "Downloads"),
                info="The finished video is saved directly here (created if it doesn't exist).",
            )
            width = gr.Number(label="Width", value=1920, precision=0)
            height = gr.Number(label="Height", value=1080, precision=0)
            fps = gr.Number(label="FPS", value=30, precision=0)
            model_size = gr.Dropdown(
                label="Speech recognition accuracy (bigger = slower, more accurate timing)",
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="small",
            )
            effects_selected = gr.Dropdown(
                label="Effects",
                choices=EFFECT_CHOICES,
                value=[ALL_OPTION] + EFFECT_CHOICES[1:],
                multiselect=True,
                # Gradio's strict choice-validation can throw on a stray transient payload
                # (e.g. rapid add/remove churn); our own _resolve_selection already ignores
                # anything unrecognized, so let it through here rather than hard-erroring.
                allow_custom_value=True,
                info="Pan/zoom effects applied randomly per scene. Deselect all to show plain static images.",
            )
            effects_prev_state = gr.State([ALL_OPTION] + EFFECT_CHOICES[1:])
            transitions_selected = gr.Dropdown(
                label="Transitions",
                choices=TRANSITION_CHOICES,
                value=[],
                multiselect=True,
                allow_custom_value=True,
                info=(
                    f"Smooth crossfade transitions applied randomly between scenes. Deselect all to "
                    f"disable and use hard cuts everywhere. Note: a transition only plays when the "
                    f"next image stays on screen for more than {MIN_DURATION_FOR_TRANSITION:.0f} seconds "
                    f"— shorter scenes always get a hard cut."
                ),
            )
            captions = gr.Checkbox(label="Burn in word-by-word karaoke captions", value=True)
            transitions_prev_state = gr.State([])

            run_btn = gr.Button("Render video", variant="primary")

            effects_selected.change(
                fn=_make_all_toggle_handler(EFFECT_CHOICES[1:]),
                inputs=[effects_selected, effects_prev_state],
                outputs=[effects_selected, effects_prev_state],
            )
            transitions_selected.change(
                fn=_make_all_toggle_handler(TRANSITION_CHOICES[1:]),
                inputs=[transitions_selected, transitions_prev_state],
                outputs=[transitions_selected, transitions_prev_state],
            )

    output_video = gr.Video(label="Result")
    log_box = gr.Textbox(label="Log", lines=12)

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            script_file, audio_file, images_zip, images_folder_path,
            width, height, fps, model_size,
            effects_selected, transitions_selected, captions,
            video_title, output_folder,
        ],
        outputs=[output_video, log_box],
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
