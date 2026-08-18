"""
Scene-Synced Slideshow Video Builder
=====================================

Turns (script + single continuous voiceover + a folder of scene-numbered images)
into a rendered MP4 where each image is shown for the EXACT duration its matching
scene is spoken in the voiceover, with rotating Ken-Burns style effects and
word-by-word karaoke captions burned in.

Everything here is free / local / open-source:
  - faster-whisper  -> speech-to-text with word-level timestamps (no API cost)
  - ffmpeg (via imageio-ffmpeg) -> all video/audio rendering
  - difflib (Python stdlib) -> aligns script words to recognized words

CLI usage:
    python pipeline.py --script script.txt --audio voiceover.mp3 \
        --images-dir ./images --out output.mp4

See README.md for setup instructions and app.py for the dashboard UI.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import difflib
import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# ffmpeg resolution (bundled binary via imageio-ffmpeg, no manual install)
# ---------------------------------------------------------------------------

def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Fall back to a system ffmpeg on PATH if imageio-ffmpeg isn't installed
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError(
                "ffmpeg not found. Install the 'imageio-ffmpeg' package "
                "(pip install imageio-ffmpeg) or put ffmpeg on your PATH."
            )
        return exe


FFMPEG = None  # resolved lazily, see _ffmpeg()


def _ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        FFMPEG = get_ffmpeg_exe()
    return FFMPEG


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n  " + " ".join(cmd) + "\n\n" + proc.stdout.decode(errors="ignore")
        )


# ---------------------------------------------------------------------------
# 1. Script -> scenes
# ---------------------------------------------------------------------------

SCENE_MARKER_RE = re.compile(
    r"^\s*(?:\[?\s*scene\s*(\d+)\s*\]?[:.\-]?|(\d+)\s*[).:]\s*)\s*(.*)$",
    re.IGNORECASE,
)

# Reasonably robust sentence splitter without extra heavy NLP dependencies.
# Splits on '.', '!', '?' followed by whitespace + capital letter/quote, or end of text.
# Avoids splitting on common abbreviations.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "st", "approx", "no", "fig", "u.s", "u.k",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    raw_parts = SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    buf = ""
    for part in raw_parts:
        buf = (buf + " " + part).strip() if buf else part
        last_word = re.sub(r"[.!?]+$", "", buf.split(" ")[-1]).lower()
        if last_word in _ABBREVIATIONS:
            continue  # keep accumulating, false split on abbreviation
        sentences.append(buf)
        buf = ""
    if buf:
        sentences.append(buf)
    return [s.strip() for s in sentences if s.strip()]


def parse_script_into_scenes(script_text: str) -> list[str]:
    """Returns an ordered list of scene texts, scene[0] == scene 1, etc."""
    lines = script_text.splitlines()
    marked: dict[int, list[str]] = {}
    any_marker = False

    current_scene = None
    for line in lines:
        if not line.strip():
            continue
        m = SCENE_MARKER_RE.match(line)
        if m:
            any_marker = True
            num = int(m.group(1) or m.group(2))
            current_scene = num
            marked.setdefault(num, [])
            rest = m.group(3).strip()
            if rest:
                marked[num].append(rest)
        elif current_scene is not None:
            marked[current_scene].append(line.strip())

    if any_marker and marked:
        max_scene = max(marked.keys())
        scenes = []
        for i in range(1, max_scene + 1):
            scenes.append(" ".join(marked.get(i, [])).strip())
        return scenes

    # No explicit scene markers found -> one sentence per scene.
    return split_sentences(script_text)


def parse_csv_into_scenes(csv_path: str) -> tuple[list[str], dict[int, str]]:
    """Alternative to parse_script_into_scenes(): reads scenes from a CSV file
    instead of a .txt script.

    Expected columns (header required, case-insensitive, a few aliases accepted):
      - scene number : "scene" / "scene_number" / "scene#" / "#" / "no"
      - text         : "text" / "script" / "narration" / "line" / "content"
      - image (opt.) : "image" / "image_path" / "img" / "file"
        If present, this overrides the numbered-filename lookup that
        find_scene_images() would otherwise do for that scene.

    Scene numbers must form a contiguous 1..N sequence (gaps raise an error,
    same rule as the marker-based script format). Rows are re-ordered by
    scene number, so the CSV rows themselves don't need to be sorted.

    Returns (scene_texts, image_overrides) where image_overrides maps
    scene_number -> path (only for scenes that had an image column filled in).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        norm_fields = {name.strip().lower(): name for name in reader.fieldnames}

        def pick(*candidates: str) -> Optional[str]:
            for c in candidates:
                if c in norm_fields:
                    return norm_fields[c]
            return None

        scene_col = pick("scene", "scene_number", "scene#", "#", "no")
        text_col = pick("text", "script", "narration", "line", "content")
        image_col = pick("image", "image_path", "img", "file")

        if scene_col is None or text_col is None:
            raise ValueError(
                "CSV must have a scene-number column (e.g. 'scene') and a "
                f"text column (e.g. 'text'). Found columns: {reader.fieldnames}"
            )

        rows: dict[int, str] = {}
        image_overrides: dict[int, str] = {}
        for row_num, row in enumerate(reader, start=2):  # header is line 1
            raw_scene = (row.get(scene_col) or "").strip()
            if not raw_scene:
                continue
            try:
                scene_num = int(raw_scene)
            except ValueError:
                raise ValueError(
                    f"CSV row {row_num}: scene number '{raw_scene}' is not an integer."
                )
            if scene_num in rows:
                raise ValueError(f"CSV row {row_num}: duplicate scene number {scene_num}.")
            rows[scene_num] = (row.get(text_col) or "").strip()
            if image_col:
                img = (row.get(image_col) or "").strip()
                if img:
                    image_overrides[scene_num] = img

    if not rows:
        raise ValueError(f"No scene rows found in CSV: {csv_path}")

    max_scene = max(rows.keys())
    missing = [i for i in range(1, max_scene + 1) if i not in rows]
    if missing:
        raise ValueError(f"CSV is missing scene number(s): {missing}")

    scenes = [rows[i] for i in range(1, max_scene + 1)]
    return scenes, image_overrides


# ---------------------------------------------------------------------------
# 2. Locate scene-numbered images
# ---------------------------------------------------------------------------

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def find_scene_images(
    images_dir: str, num_scenes: int, overrides: Optional[dict[int, str]] = None
) -> dict[int, str]:
    images_dir = Path(images_dir)
    by_scene: dict[int, str] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = re.search(r"(\d+)$", path.stem)
        if not match:
            continue
        scene_num = int(match.group(1))
        by_scene.setdefault(scene_num, str(path))

    # CSV-supplied explicit image paths win over the numbered-filename lookup.
    if overrides:
        for scene_num, img_ref in overrides.items():
            candidates = [Path(img_ref), images_dir / img_ref]
            resolved = next((str(c) for c in candidates if c.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"CSV image override for scene {scene_num} not found "
                    f"(tried '{img_ref}' and '{images_dir / img_ref}')."
                )
            by_scene[scene_num] = resolved

    missing = [i for i in range(1, num_scenes + 1) if i not in by_scene]
    if missing:
        raise FileNotFoundError(
            f"Not enough scene images in {images_dir}\n"
            f"Scenes Found: {num_scenes}\n"
            f"Images Found: {len(by_scene)}\n"
            f"Missing scene number(s): {missing}"
        )
    return {i: by_scene[i] for i in range(1, num_scenes + 1)}


# ---------------------------------------------------------------------------
# 3. Transcribe voiceover with word-level timestamps (faster-whisper)
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe_words(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    total_duration: Optional[float] = None,
    progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
) -> list[Word]:
    from faster_whisper import WhisperModel

    if progress_cb:
        progress_cb(f"Loading whisper model '{model_size}'...", 0.0)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if progress_cb:
        progress_cb("Transcribing voiceover (this can take a few minutes)...", 0.02)
    segments, _info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)

    words: list[Word] = []
    for seg in segments:
        if progress_cb and total_duration:
            frac = min(0.98, 0.02 + 0.98 * (seg.end / total_duration))
            progress_cb(f"Transcribing... {seg.end:.0f}s / {total_duration:.0f}s", frac)
        if not seg.words:
            continue
        for w in seg.words:
            if w.word.strip():
                words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
    return words


# ---------------------------------------------------------------------------
# 4. Align script scenes to the transcribed word timeline
# ---------------------------------------------------------------------------

def _normalize(token: str) -> str:
    return re.sub(r"[^a-z0-9']", "", token.lower())


def align_scenes(
    scenes: list[str], words: list[Word], total_audio_duration: float
) -> list[tuple[float, float]]:
    """
    Returns a list of (start_time, end_time) per scene, covering the full
    audio duration with no gaps (scene N end == scene N+1 start), because the
    narrator is presumed to read the script continuously start to finish.
    """
    # Flatten script into a single normalized token stream, remembering which
    # scene each token index belongs to.
    script_tokens: list[str] = []
    token_scene: list[int] = []
    for scene_idx, scene_text in enumerate(scenes):
        for tok in scene_text.split():
            norm = _normalize(tok)
            if norm:
                script_tokens.append(norm)
                token_scene.append(scene_idx)

    whisper_tokens = [_normalize(w.text) for w in words]

    sm = difflib.SequenceMatcher(a=script_tokens, b=whisper_tokens, autojunk=False)
    matches = sm.get_matching_blocks()  # list of Match(a, b, size), ends with a zero-size sentinel

    # Build anchors: script_token_index -> whisper word start time (for block start)
    # and -> whisper word end time (for block end), only for tokens inside matched blocks.
    anchor_positions: list[tuple[int, float]] = []  # (script_index, timestamp)
    for m in matches:
        if m.size == 0:
            continue
        for k in range(m.size):
            script_idx = m.a + k
            whisper_idx = m.b + k
            w = words[whisper_idx]
            # use midpoint of the matched word as the anchor timestamp
            anchor_positions.append((script_idx, (w.start + w.end) / 2))

    if not anchor_positions:
        # Total fallback: no words matched at all (e.g. transcription failed) ->
        # split audio evenly across scenes so the tool still produces output.
        n = len(scenes)
        step = total_audio_duration / max(n, 1)
        return [(i * step, (i + 1) * step) for i in range(n)]

    anchor_positions.sort(key=lambda p: p[0])
    anchor_idx = [p[0] for p in anchor_positions]
    anchor_time = [p[1] for p in anchor_positions]

    def timestamp_for_script_index(idx: int) -> float:
        # Binary-search-free simple interpolation (script is short enough that
        # linear scan is fine; anchor_idx is sorted and small relative to a 20 min video)
        if idx <= anchor_idx[0]:
            return max(0.0, anchor_time[0])
        if idx >= anchor_idx[-1]:
            return min(total_audio_duration, anchor_time[-1])
        # find bracketing anchors
        lo, hi = 0, len(anchor_idx) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if anchor_idx[mid] <= idx:
                lo = mid
            else:
                hi = mid
        i0, i1 = anchor_idx[lo], anchor_idx[hi]
        t0, t1 = anchor_time[lo], anchor_time[hi]
        if i1 == i0:
            return t0
        frac = (idx - i0) / (i1 - i0)
        return t0 + frac * (t1 - t0)

    # scene boundaries = first token index of each scene
    scene_start_token: list[int] = []
    for scene_idx in range(len(scenes)):
        try:
            first_tok = token_scene.index(scene_idx)
        except ValueError:
            first_tok = None
        scene_start_token.append(first_tok)

    n = len(scenes)
    raw_starts: list[float] = [0.0] * n
    for i in range(n):
        tok = scene_start_token[i]
        if tok is None:
            raw_starts[i] = raw_starts[i - 1] if i > 0 else 0.0
        else:
            raw_starts[i] = timestamp_for_script_index(tok)

    raw_starts[0] = 0.0
    # enforce monotonic non-decreasing boundaries
    for i in range(1, n):
        if raw_starts[i] < raw_starts[i - 1]:
            raw_starts[i] = raw_starts[i - 1]

    boundaries = raw_starts + [total_audio_duration]
    return [(boundaries[i], boundaries[i + 1]) for i in range(n)]


# ---------------------------------------------------------------------------
# 5. Render each scene clip with a Ken-Burns style effect, exact duration
# ---------------------------------------------------------------------------

EFFECTS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]

EFFECT_LABELS = {
    "zoom_in": "Zoom In",
    "zoom_out": "Zoom Out",
    "pan_left": "Pan Left",
    "pan_right": "Pan Right",
    "pan_up": "Pan Up",
    "pan_down": "Pan Down",
}

# Curated subset of ffmpeg's xfade transitions: smooth crossfade/wipe styles only,
# excluding the geometric/pixelation ones (pixelize, radial, wind, slice, squeeze, ...)
# that read as heavy or gimmicky on a slideshow.
TRANSITIONS = [
    "fade", "fadeblack", "fadewhite", "dissolve",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
]

TRANSITION_LABELS = {
    "fade": "Fade",
    "fadeblack": "Fade to Black",
    "fadewhite": "Fade to White",
    "dissolve": "Dissolve",
    "smoothleft": "Smooth Left",
    "smoothright": "Smooth Right",
    "smoothup": "Smooth Up",
    "smoothdown": "Smooth Down",
}

# A transition is only applied between two scenes if the next scene's on-screen
# duration exceeds this many seconds; shorter scenes get a hard cut instead.
MIN_DURATION_FOR_TRANSITION = 5.0

# Below this duration, a Ken-Burns pan/zoom effect barely gets going before the
# scene cuts away -- it reads as a jerky blip rather than a deliberate motion.
# Scenes shorter than this show the plain static image instead (same reasoning
# as MIN_DURATION_FOR_TRANSITION, just for effects instead of transitions).
MIN_DURATION_FOR_EFFECT = 4.0


def _zoompan_filter(effect: str, duration: float, fps: int, out_w: int, out_h: int) -> str:
    frames = max(1, round(duration * fps))
    # Zoom travels from 1.0x to 1.5x (or back). Scaling the per-frame rate to
    # this scene's frame count means it always lands on 1.5x/1.0x exactly on
    # the LAST frame, no matter how long the scene is. A fixed rate (e.g.
    # always 0.0015/frame) reaches the cap after a fixed ~11s regardless of
    # scene length -- for any scene longer than that, the image visibly
    # freezes (zoom stuck at the cap) for the remaining duration while the
    # voiceover keeps playing.
    zoom_range = 0.5  # 1.0 -> 1.5
    zoom_rate = zoom_range / frames

    if effect == "zoom_in":
        z = f"min(zoom+{zoom_rate:.8f},1.5)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif effect == "zoom_out":
        z = f"if(eq(on,1),1.5,max(zoom-{zoom_rate:.8f},1.0))"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif effect == "pan_left":
        z = "1.2"
        x, y = f"(iw-iw/zoom)*(1-on/{frames})", "ih/2-(ih/zoom/2)"
    elif effect == "pan_right":
        z = "1.2"
        x, y = f"(iw-iw/zoom)*(on/{frames})", "ih/2-(ih/zoom/2)"
    elif effect == "pan_up":
        z = "1.2"
        x, y = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*(1-on/{frames})"
    else:  # pan_down
        z = "1.2"
        x, y = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*(on/{frames})"

    return (
        f"scale=2560:-1,zoompan=z='{z}':d={frames}:x='{x}':y='{y}':"
        f"s={out_w}x{out_h}:fps={fps},format=yuv420p"
    )


def render_scene_clip(
    image_path: str,
    duration: float,
    effect: Optional[str],
    out_path: str,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    ass_path: Optional[str] = None,
) -> None:
    out_w, out_h = resolution
    if effect:
        vf = _zoompan_filter(effect, duration, fps, out_w, out_h)
    else:
        vf = (
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},format=yuv420p"
        )
    if ass_path:
        # Burn captions in the same filter chain/encode pass instead of a second
        # ffmpeg call over the already-encoded clip -- halves the encode work per scene.
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")
        vf = f"{vf},subtitles='{ass_escaped}'"
    cmd = [
        _ffmpeg(), "-y",
        # -framerate must match the output fps: without it, ffmpeg assumes a default
        # 25fps clock for the looped still image, so -t duration only ever supplies
        # duration*25 frames. At fps=30 that's close enough to pass unnoticed; at
        # fps=60 zoompan's d=frames (duration*60) needs more frames than the input
        # can supply within -t, so ffmpeg hits EOF early and the clip renders short
        # -- the shortfall compounds across scenes into audio/video drift.
        "-loop", "1", "-framerate", str(fps), "-i", image_path,
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-r", str(fps),
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        out_path,
    ]
    run(cmd)


# ---------------------------------------------------------------------------
# 6. Word-by-word karaoke captions (ASS subtitles, burned in per scene)
# ---------------------------------------------------------------------------

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial Black,{font_size},&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def build_scene_ass(
    scene_words: list[Word], scene_start: float, scene_end: float, out_path: str,
    resolution: tuple[int, int] = (1920, 1080),
) -> None:
    out_w, out_h = resolution
    font_size = max(28, out_h // 18)
    lines = [ASS_HEADER.format(res_x=out_w, res_y=out_h, font_size=font_size)]

    if scene_words:
        karaoke_parts = []
        for w in scene_words:
            dur_cs = max(1, round((w.end - w.start) * 100))
            karaoke_parts.append(f"{{\\k{dur_cs}}}{w.text}")
        text = " ".join(karaoke_parts)
        start_rel = 0.0
        end_rel = max(0.1, scene_end - scene_start)
        lines.append(
            f"Dialogue: 0,{_ass_time(start_rel)},{_ass_time(end_rel)},Karaoke,,0,0,0,,{text}\n"
        )

    Path(out_path).write_text("".join(lines), encoding="utf-8")


def burn_subtitles(video_in: str, ass_path: str, video_out: str) -> None:
    ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        _ffmpeg(), "-y", "-i", video_in,
        "-vf", f"subtitles='{ass_escaped}'",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        video_out,
    ]
    run(cmd)


# ---------------------------------------------------------------------------
# 6b. Crossfade transitions between scenes (ffmpeg xfade), where eligible
# ---------------------------------------------------------------------------

def _render_xfade_group(
    group: list[tuple[str, float]],
    transitions: list[str],
    rng: random.Random,
    out_path: str,
    fps: int,
) -> None:
    """Chain-merges clips in `group` with random crossfade transitions via xfade."""
    cmd = [_ffmpeg(), "-y"]
    for path, _ in group:
        cmd += ["-i", path]

    length_before = group[0][1]
    cur_label = "0:v"
    filter_chain = []
    for idx in range(1, len(group)):
        dur_next = group[idx][1]
        name = rng.choice(transitions)
        # Bounded only by dur_next: the incoming clip must have this much real content
        # to blend. The outgoing side no longer needs a matching bound (see pad below).
        xfade_dur = max(0.08, min(0.6, dur_next) - 0.05)

        # Pad the growing chain's tail with cloned frozen frames before crossfading,
        # so the overlap consumes newly-added time instead of cannibalizing real scene
        # content. Without this, each transition shrinks total video duration by
        # xfade_dur, and since scene timing is derived from the voiceover, that
        # shrinkage compounds across every transition -- the video runs increasingly
        # ahead of the audio the more transitions it passes through.
        padded_label = f"pad{idx}"
        out_label = f"v{idx}"
        filter_chain.append(
            f"[{cur_label}]tpad=stop_duration={xfade_dur:.3f}:stop_mode=clone[{padded_label}]"
        )
        filter_chain.append(
            f"[{padded_label}][{idx}:v]xfade=transition={name}:duration={xfade_dur:.3f}:"
            f"offset={length_before:.3f}[{out_label}]"
        )
        length_before = length_before + dur_next
        cur_label = out_label

    cmd += [
        "-filter_complex", ";".join(filter_chain),
        "-map", f"[{cur_label}]",
        "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        out_path,
    ]
    run(cmd)


def apply_transitions(
    clip_specs: list[tuple[str, float]],
    transitions: Optional[list[str]],
    rng: random.Random,
    workdir: str,
    fps: int,
) -> list[str]:
    """Groups consecutive clips into crossfade chains, splitting at any boundary
    where the next scene's duration doesn't clear MIN_DURATION_FOR_TRANSITION.
    Returns a flat list of clip paths ready for hard-concat."""
    if not transitions:
        return [path for path, _ in clip_specs]

    groups: list[list[tuple[str, float]]] = [[clip_specs[0]]]
    for i in range(1, len(clip_specs)):
        _, next_duration = clip_specs[i]
        if next_duration > MIN_DURATION_FOR_TRANSITION:
            groups[-1].append(clip_specs[i])
        else:
            groups.append([clip_specs[i]])

    result_paths = []
    for gi, group in enumerate(groups):
        if len(group) == 1:
            result_paths.append(group[0][0])
        else:
            out_path = os.path.join(workdir, f"xfade_group_{gi:04d}.mp4")
            _render_xfade_group(group, transitions, rng, out_path, fps)
            result_paths.append(out_path)
    return result_paths


# ---------------------------------------------------------------------------
# 7. Concat scene clips + mux original voiceover
# ---------------------------------------------------------------------------

def concat_clips(clip_paths: list[str], out_path: str, workdir: str) -> None:
    list_path = os.path.join(workdir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        _ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", out_path,
    ]
    run(cmd)


def mux_audio(video_in: str, audio_in: str, out_path: str) -> None:
    cmd = [
        _ffmpeg(), "-y", "-i", video_in, "-i", audio_in,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    run(cmd)


def get_audio_duration(audio_path: str) -> float:
    cmd = [
        _ffmpeg(), "-i", audio_path, "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode(errors="ignore")
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", out)
    if not m:
        raise RuntimeError("Could not determine audio duration from ffmpeg output.")
    h, mi, s = m[-1]
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _render_one_scene(args: tuple) -> tuple[int, str, float]:
    """Worker for parallel scene rendering. Must stay a top-level function (not a
    closure/method) so it can be pickled and sent to a separate process."""
    (i, image_path, duration, effect, scene_words, captions,
     resolution, fps, workdir) = args

    ass_path = None
    if captions:
        ass_path = os.path.join(workdir, f"scene_{i:04d}.ass")
        build_scene_ass(scene_words, 0.0, duration, ass_path, resolution=resolution)

    scene_clip = os.path.join(workdir, f"scene_{i:04d}.mp4")
    render_scene_clip(
        image_path, duration, effect, scene_clip,
        resolution=resolution, fps=fps, ass_path=ass_path,
    )
    return i, scene_clip, duration


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_video(
    audio_path: str,
    images_dir: str,
    out_path: str,
    script_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    model_size: str = "small",
    captions: bool = True,
    effects: Optional[list[str]] = None,
    effect_weights: Optional[dict[str, float]] = None,
    transitions: Optional[list[str]] = None,
    seed: int = 42,
    progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
) -> str:
    """
    script_path / csv_path: exactly one must be given. script_path is a .txt
        file parsed by parse_script_into_scenes(); csv_path is a CSV file
        parsed by parse_csv_into_scenes() (columns: scene, text, and
        optionally image -- see that function's docstring). Both produce the
        same ordered list of scene texts feeding the rest of the pipeline.
    effects: pool of Ken-Burns effect names to apply randomly per scene.
        None -> use all of EFFECTS (default/back-compat). [] -> no effect at all.
    effect_weights: optional {effect_name: weight} mix controlling how often each
        effect in `effects` is picked (e.g. {"zoom_in": 70, "pan_left": 30} makes
        zoom_in ~2.3x more likely than pan_left). Weights are relative, not
        required to sum to 100. Effects not present in `effects` are ignored even
        if a weight is given for them. None -> effects are cycled evenly
        (original round-robin shuffle behavior). An effect present in `effects`
        but missing from effect_weights (or given weight 0) never gets picked.
    transitions: pool of xfade transition names to apply randomly between eligible
        scenes. None or [] -> no transitions (hard cuts only).

    progress_cb, if given, is called as progress_cb(message, fraction) where
    fraction is a float in [0, 1] estimating overall completion (transcription
    and per-scene rendering are the dominant phases and are weighted as such).
    """
    def log(msg: str, frac: Optional[float] = None) -> None:
        if progress_cb:
            progress_cb(msg, frac)
        print(msg)

    if bool(script_path) == bool(csv_path):
        raise ValueError("Provide exactly one of script_path or csv_path (not both, not neither).")

    image_overrides: dict[int, str] = {}
    if csv_path:
        scenes, image_overrides = parse_csv_into_scenes(csv_path)
        log(f"Parsed {len(scenes)} scene(s) from CSV.", 0.01)
    else:
        script_text = Path(script_path).read_text(encoding="utf-8")
        scenes = parse_script_into_scenes(script_text)
        log(f"Parsed {len(scenes)} scene(s) from script.", 0.01)

    scene_images = find_scene_images(images_dir, len(scenes), overrides=image_overrides)
    log("All scene images found.", 0.02)

    total_duration = get_audio_duration(audio_path)
    log(f"Voiceover duration: {total_duration:.2f}s", 0.03)

    def transcribe_progress(msg: str, sub_frac: Optional[float]) -> None:
        frac = 0.03 + (sub_frac or 0.0) * 0.50  # transcription: 3% -> 53%
        log(msg, frac)

    words = transcribe_words(
        audio_path, model_size=model_size, total_duration=total_duration,
        progress_cb=transcribe_progress,
    )
    log(f"Transcribed {len(words)} words with timestamps.", 0.53)

    boundaries = align_scenes(scenes, words, total_duration)
    log("Aligned scenes to the audio timeline.", 0.55)

    rng = random.Random(seed)
    effect_pool = EFFECTS[:] if effects is None else effects[:]

    if effect_weights:
        # Weighted mode: only effects that are both in the pool AND carry a
        # positive weight are eligible; each scene independently draws from
        # that weighted distribution.
        weighted_effects = [e for e in effect_pool if effect_weights.get(e, 0) > 0]
        weights = [effect_weights[e] for e in weighted_effects]
        if weighted_effects:
            def next_effect() -> Optional[str]:
                return rng.choices(weighted_effects, weights=weights, k=1)[0]
        else:
            def next_effect() -> Optional[str]:
                return None
    else:
        # Back-compat mode: cycle evenly through a shuffled pool.
        effect_order = effect_pool[:]
        rng.shuffle(effect_order)

    with tempfile.TemporaryDirectory(prefix="scenevid_") as workdir:
        # Build the per-scene task list up front, then render all scenes in
        # parallel worker processes -- each ffmpeg render is CPU-bound and
        # independent of the others, so this scales with available cores
        # instead of rendering one scene at a time.
        tasks = []
        for i, (scene_text, (start, end)) in enumerate(zip(scenes, boundaries), start=1):
            duration = max(0.05, end - start)
            if effect_weights:
                effect = next_effect()
            else:
                effect = effect_order[(i - 1) % len(effect_order)] if effect_order else None
            if duration < MIN_DURATION_FOR_EFFECT:
                effect = None  # too short for a pan/zoom to read as intentional
            scene_words = [w for w in words if start <= (w.start + w.end) / 2 < end] if captions else []
            tasks.append((
                i, scene_images[i], duration, effect, scene_words,
                captions, resolution, fps, workdir,
            ))

        max_workers = min(len(tasks), os.cpu_count() or 1)
        results: dict[int, tuple[str, float]] = {}
        done_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_render_one_scene, t): t[0] for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                i, scene_clip, duration = future.result()
                results[i] = (scene_clip, duration)
                done_count += 1
                frac = 0.55 + (done_count / len(tasks)) * 0.35  # per-scene rendering: 55% -> 90%
                log(f"Scene {i}/{len(scenes)} rendered ({done_count}/{len(tasks)} done)", frac)

        # Recombine in original scene order (parallel completion order is arbitrary).
        clip_specs: list[tuple[str, float]] = [results[i] for i in range(1, len(scenes) + 1)]

        if transitions:
            log("Applying transitions between eligible scenes...", 0.90)
        clip_paths = apply_transitions(clip_specs, transitions, rng, workdir, fps)

        log("Concatenating scene clips...", 0.93)
        concat_video = os.path.join(workdir, "concat_video.mp4")
        concat_clips(clip_paths, concat_video, workdir)

        log("Muxing original voiceover audio...", 0.97)
        mux_audio(concat_video, audio_path, out_path)

    log(f"Done -> {out_path}", 1.0)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--script", help="Path to script .txt file")
    source_group.add_argument(
        "--csv",
        help="Path to CSV file with scene,text[,image] columns (alternative to --script)",
    )
    ap.add_argument("--audio", required=True, help="Path to voiceover audio file")
    ap.add_argument("--images-dir", required=True, help="Folder containing 1.png, 2.jpg, ... per scene")
    ap.add_argument("--out", required=True, help="Output .mp4 path")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    ap.add_argument("--no-captions", action="store_true")
    args = ap.parse_args()

    build_video(
        script_path=args.script,
        csv_path=args.csv,
        audio_path=args.audio,
        images_dir=args.images_dir,
        out_path=args.out,
        resolution=(args.width, args.height),
        fps=args.fps,
        model_size=args.model,
        captions=not args.no_captions,
    )


if __name__ == "__main__":
    # Same reason as app.py: build_video() uses ProcessPoolExecutor, and if this
    # script is ever run from a frozen/PyInstaller .exe (instead of only being
    # imported by app.py), every spawned worker process needs this call first to
    # avoid re-executing the whole CLI from scratch. Harmless as a no-op on a
    # normal `python pipeline.py ...` run.
    multiprocessing.freeze_support()
    main()
