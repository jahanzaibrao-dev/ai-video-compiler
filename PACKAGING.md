# Packaging as a one-click desktop app (no Python required for the end user)

PyInstaller bundles Python itself, all dependencies (including the ffmpeg binary
pulled in by `imageio-ffmpeg`), and your code into a single folder/executable.
The person you give it to just double-clicks it — the dashboard opens in their
browser automatically, nothing to install.

**Important constraint:** PyInstaller is not a cross-compiler. Build on Windows
to get a `.exe`, build on Mac to get a `.app`. If you need both, run the build
step on one machine of each OS (or ask me to do it in a matching sandbox).

## One-time build steps

```bash
# Inside your project folder, with the venv from README.md activated
pip install pyinstaller

pyinstaller --name "SceneVideoBuilder" \
  --onedir \
  --add-data "app.py:." \
  --collect-all faster_whisper \
  --collect-all imageio_ffmpeg \
  --collect-all gradio \
  --collect-all gradio_client \
  app.py
```

Notes on the flags:
- `--onedir` (not `--onefile`) is recommended here — Whisper model files and
  ffmpeg binaries are large, and `--onedir` starts up much faster than
  `--onefile`, which has to unpack everything to a temp folder on every launch.
- `--collect-all faster_whisper` and `--collect-all imageio_ffmpeg` make sure
  the ffmpeg binary and whisper package data get copied into the bundle.
- `--collect-all gradio` / `gradio_client` is needed because Gradio ships
  static frontend assets (JS/CSS) that PyInstaller doesn't pick up automatically.

The build lands in `dist/SceneVideoBuilder/`. Zip that whole folder — that's
the thing you hand to someone. They unzip it and double-click
`SceneVideoBuilder.exe` (Windows) or `SceneVideoBuilder` (Mac/Linux).

## Whisper model: bundle it or download-on-first-run?

By default the whisper model downloads from Hugging Face the first time the
app runs (needs internet once, then it's cached). To make the package fully
offline-capable instead:

1. Run `python pipeline.py --model small ...` once yourself so the model
   downloads into your Hugging Face cache folder
   (`~/.cache/huggingface` on Mac/Linux, `%USERPROFILE%\.cache\huggingface` on Windows).
2. Copy that cache folder into your PyInstaller `dist/SceneVideoBuilder/` output.
3. In `pipeline.py`, when constructing `WhisperModel(...)`, add
   `download_root="<bundled cache path>"` so it looks there first.

This adds ~150-500MB to the package size (depending on model size chosen) but
means zero internet dependency for the end user.

## Quick sanity check before shipping

After building, on a clean machine (or a fresh user account) with no Python
installed:
1. Unzip the dist folder.
2. Double-click the executable.
3. Confirm the browser dashboard opens and a short test render (small script,
   3-4 scenes, a few seconds of audio) completes and downloads correctly.

## Alternative: skip packaging, ship a one-line installer script instead

If building native executables for multiple OSes becomes a maintenance burden,
a lighter-weight alternative that still requires near-zero setup: ship a single
`install_and_run.bat` (Windows) / `install_and_run.sh` (Mac/Linux) script that
checks for Python, installs it if missing (via `winget`/`brew`), creates the
venv, installs `requirements.txt`, and launches `app.py`. First run takes a
few minutes to set up; every run after that is instant. This avoids the
platform-specific build step entirely and keeps update distribution as simple
as re-sharing the project folder.
