# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, free, no-API tool that turns (script + one continuous voiceover + a folder of
scene-numbered images) into a rendered MP4, where each image is shown for the exact
duration its scene is spoken, with rotating Ken-Burns pan/zoom effects, optional
crossfade transitions, and word-by-word karaoke captions burned in. Two files:
`pipeline.py` (all video-processing logic, no UI) and `app.py` (Gradio dashboard wrapper
around it). See `README.md` for user-facing setup/usage and `REPORT.md` for why this
approach (vs. existing tools) was chosen.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the dashboard (opens a browser tab at 127.0.0.1:7860)
python app.py

# Run headless via CLI
python pipeline.py --script script.txt --audio voiceover.mp3 \
    --images-dir ./images --out output.mp4
# optional: --width --height --fps --model {tiny,base,small,medium,large-v3} --no-captions
```

No test suite or linter is configured in this repo. Verification has historically been
done by running `python app.py` (or driving it headlessly with Playwright) and by
probing rendered clips with `ffmpeg -i <file>` / frame-count checks — not pytest.

**Packaging (Windows .exe / Mac .app via PyInstaller):** see `PACKAGING.md` for the
full command and the non-obvious `--collect-all` flags it requires. There's also
`.github/workflows/build-windows-exe.yml` — manual-trigger only (`workflow_dispatch`
with a `version` input), builds on `windows-latest`, publishes a GitHub Release with
the zipped build attached. Keep the PyInstaller command in the workflow and in
`PACKAGING.md` in sync if either changes.

## Architecture

`pipeline.py`'s `build_video()` is the orchestrator; everything else in the file is a
numbered stage it calls in order:

1. **Script → scenes** (`parse_script_into_scenes`). Explicit `Scene 1:` / `[Scene 2]` /
   `1)` markers if present; otherwise falls back to one-sentence-per-scene
   (`split_sentences`). This fallback is why a script with no markers can produce far
   more scenes than expected — scene *count* is not knowable without actually parsing.
2. **Locate images** (`find_scene_images`). Matches files by the *trailing digit run* in
   the filename stem (regex), not an exact `f"{i}.png"` string — so `1.png`, `001.png`,
   `0001.png`, and even `scene_001.png` all resolve to scene 1. Don't regress this to
   exact-string matching.
3. **Transcribe** (`transcribe_words`, faster-whisper, word-level timestamps, CPU/int8
   by default).
4. **Align** (`align_scenes`). Uses `difflib.SequenceMatcher` to match normalized script
   tokens against normalized transcribed words (handles minor ASR errors), then
   interpolates a timestamp for each scene's first token. Scene boundaries are forced
   contiguous (scene N's end == scene N+1's start) and monotonic — there are no gaps,
   because the narrator is presumed to read straight through.
5. **Render each scene clip** (`render_scene_clip` + `_zoompan_filter`). One ffmpeg call
   per scene: loops the still image for exactly `duration` seconds with a Ken-Burns
   `zoompan` filter (or a plain scale/crop if `effect=None`). The looped image's
   `-framerate` is explicitly pinned to match the output `fps` — without that, ffmpeg
   assumes a default input clock and `zoompan`'s `d=frames` (duration×fps) can run out
   of source frames at high fps, silently rendering the clip short. Don't drop that flag.
6. **Captions** (`build_scene_ass` + `burn_subtitles`). Per-scene ASS karaoke subtitles,
   burned in with a second ffmpeg pass.
7. **Transitions** (`apply_transitions` + `_render_xfade_group`). Groups consecutive
   scenes into `xfade` crossfade chains, but only where the *next* scene's duration
   exceeds `MIN_DURATION_FOR_TRANSITION` (5s) — shorter scenes always hard-cut. Each
   crossfade pads the growing chain's tail with cloned frozen frames (`tpad`) before
   blending, so the overlap consumes newly-added time rather than cannibalizing real
   scene duration. This is load-bearing: without the pad, every transition shrinks total
   video length, and since scene timing is derived from the voiceover, that shrinkage
   compounds across every transition into audio/video drift that gets worse toward the
   end of the video. If you touch this function, verify total output duration still
   equals the sum of scene durations (see conversation history / test methodology).
8. **Concat + mux** (`concat_clips`, `mux_audio`). Stream-copy concat (fast, no
   re-encode) of whatever `apply_transitions` produced, then mux the original voiceover
   in with `-shortest`.

`progress_cb(message, fraction)` is threaded through every stage as a two-argument
callback (fraction is a float 0–1, phase-weighted: setup 0–3%, transcription 3–53%,
alignment 55%, per-scene rendering 55–90%, transitions/concat/mux 90–100%). Both
arguments are required wherever this callback is invoked or accepted.

`effects`/`transitions` params on `build_video()`: `None` means "use the full default
pool" (CLI back-compat); `[]` explicitly means "disabled" (no effect / no transitions).
These are not interchangeable — preserve the distinction.

### `app.py` (Gradio dashboard)

Thin wrapper: collects inputs, resolves the Effects/Transitions multiselect UI state
into a `list[str]` via `_resolve_selection`, and calls `build_video()`. Two things here
are non-obvious Gradio footguns, already fixed — don't reintroduce them:

- The Effects/Transitions "All" pseudo-option is kept in sync with individual
  selections via a `.change()` handler that writes back to the *same* component. That
  write itself fires another `.change()` event; without the `gr.skip()` guard in
  `_make_all_toggle_handler` when no correction is actually needed, this becomes an
  infinite self-retriggering loop (confirmed empirically: ~2200 handler invocations in
  15 seconds before the fix). Any new `.change()` handler that writes back to its own
  trigger component needs the same no-op guard.
- The rendered video is now written directly to a user-chosen `output_folder` (not a
  temp dir), so it's outside Gradio's default allowed-serving paths. `demo.allowed_paths`
  is appended to at request time (inside `run_pipeline`) once the folder is known —
  `launch(allowed_paths=...)` can't be used since the folder isn't known until runtime.
