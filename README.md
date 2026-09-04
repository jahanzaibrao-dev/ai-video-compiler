# Scene-Synced Slideshow Video Builder

Local, free, no-API tool that turns a script + a single continuous voiceover + a
folder of scene-numbered images and/or video clips into a rendered MP4, with each
image or clip shown for the *exact* duration its scene is spoken, rotating
Ken-Burns pan/zoom effects on image scenes, and word-by-word karaoke captions
burned in.

See `REPORT.md` for the full research writeup (existing tools compared, why this
approach was chosen). This file covers setup and day-to-day use.

## How it works (short version)

1. `pipeline.py` splits your script into scenes (explicit `Scene 1:` markers if
   present, otherwise one sentence = one scene).
2. It transcribes your voiceover locally with `faster-whisper` to get a
   word-by-word timestamp map — no internet/API calls after the model downloads once.
3. It aligns your script's words to the transcript (same technique as a text diff)
   to find exactly when each scene starts/ends in the audio.
4. It renders each `<scene_number>.png/jpg/mp4/...` for that exact duration —
   images get a rotating pan/zoom effect, video clips play as-is (looped/trimmed
   to fit) — burns in synced karaoke captions, then stitches everything together
   and drops your original voiceover on top.

## 1. First-time setup

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No separate ffmpeg install needed — `imageio-ffmpeg` downloads a static ffmpeg
binary automatically the first time it's used.

The first run will also download the whisper speech-recognition model
(~150MB for the default "small" model) from Hugging Face — this happens once
and is cached locally after that.

## 2. Prepare your inputs

- `script.txt` — your full script. Either:
  - Mark scenes explicitly, one per line, e.g.:
    ```
    Scene 1: A quiet street at dawn.
    Scene 2: The old man opens his shop.
    ```
  - Or just write normal prose — each sentence becomes a scene automatically
    (sentence 1 → scene 1, etc.), matching your stated requirement.
- `voiceover.mp3` (or .wav/.m4a) — the single continuous narration of the whole script.
- `images/` folder — `1.png` (or .jpg), `2.png`, `3.jpg`, ... one per scene number.
  A scene's source can also be a video clip instead of a still image — e.g.
  `4.mp4` — named the same way (`<scene_number>.ext`); supported video
  extensions are `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.m4v`. Video scenes
  are scaled/cropped to fill the frame and looped or trimmed to the scene's
  exact duration, but don't get the Ken-Burns pan/zoom effect (they already
  have real motion), and any audio embedded in the clip is dropped — only the
  single continuous voiceover is heard.

## 3. Run it

**Dashboard (recommended):**

```bash
python app.py
```

Opens a browser tab where you upload the script/audio and point to the
images/video-clips folder (or upload a zip of them), pick resolution/fps/caption
toggle, and click Render. Good for 200-300 files since you just point to a
folder path rather than uploading each file individually.

**Command line:**

```bash
python pipeline.py --script script.txt --audio voiceover.mp3 \
    --images-dir ./images --out output.mp4
```

Optional flags: `--width`, `--height`, `--fps`, `--model` (tiny/base/small/medium/large-v3 —
bigger = more accurate scene-timing, slower), `--no-captions`.

## 4. Accuracy notes on "exact" scene duration

Timing is accurate to a single video frame (e.g. 1/30th of a second at 30fps) —
as strict as digital video allows. The word-timestamp accuracy of the underlying
speech model is typically within ~50-100ms on clear narration audio. If you need
tighter word-level caption precision, swap `faster-whisper` for `WhisperX`
(adds a forced-alignment pass with a phoneme model) — see the note in `REPORT.md`;
it's a heavier dependency (~2GB, needs PyTorch) so it's not the default here.

## 5. Packaging as a one-click desktop app

See `PACKAGING.md`.
