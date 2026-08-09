# Scene-Synced Slideshow Video Builder — Research & Recommendation

## The core technical problem

Your requirement isn't really "combine images and audio" — that part is trivial. The hard part is this: you have **one continuous voiceover file** for the whole script, but you need to know the **exact millisecond** each scene's narration starts and ends inside that single file, so the matching image shows for *exactly* that long. Since you're not recording separate audio per scene, no tool can read that boundary from metadata — it has to be figured out from the audio itself.

There are only two honest ways to get that:

1. **Speech-to-text with word-level timestamps**, then match the recognized words back to your script text to find where each scene's sentences start/end. (What I recommend — free, local, ~1 frame accurate.)
2. **Silence/pause detection** between scenes, which only works if your narrator pauses cleanly between every scene and never pauses mid-scene. Fragile in practice — misses are common on 15-20 min scripts with 200+ scenes.

Everything below is built around option 1, using **faster-whisper** (a free, local, open-source speech recognizer) to get word timestamps, then aligning those words back to your script text.

## Existing free tools — evaluated against your exact requirements

| Tool | Free? | Handles scene-numbered image sequencing | Exact per-scene duration from one voiceover | Local/no API cost | Effects + captions | Verdict |
|---|---|---|---|---|---|---|
| **CapCut / VN / DaVinci Resolve** (manual editors) | Yes | No — manual drag-drop | No — you'd trim every clip by hand (200-300 times) | Local | Yes, great effects | Great editor, zero automation. Not what you need for bulk. |
| **Pictory / InVideo / Fliki** (AI video SaaS) | Free tier only, then paid | No — they auto-pick stock images/AI images, don't accept your labeled image set as scene-mapped input | No exact control over your specific requirement | Cloud, paid API tiers for volume | Yes | Wrong shape of tool — built for auto-sourcing visuals, not for placing *your* pre-made, pre-numbered image set. |
| **JSON2Video / Shotstack / Creatomate** (video-generation APIs) | No — usage-based paid API | Yes, if you script the JSON timeline yourself | Yes, if you compute per-scene durations yourself and pass them in | Cloud API (cost per render) | Yes | Could work as the *rendering* layer, but you still have to build the exact same alignment logic yourself, and now you're paying per video for something you can run free, locally, in ffmpeg. |
| **Remotion** (React-based video-as-code) | Yes (Apache-2.0 core) | Yes, fully — you write the timeline in code | Yes, if you feed it computed scene durations | Local rendering, no per-video API cost | Yes, very high quality effects (it's real CSS/React) | Genuinely viable, but requires a Node/React dev environment. Heavier learning curve and packaging story for a non-technical end user than a Python+ffmpeg tool. |
| **VUZA / MoneyPrinterTurbo / similar open-source "faceless video" repos** (GitHub) | Yes | Partially — they usually auto-fetch or auto-generate images per line, not designed to consume *your own numbered image folder* as authoritative scene assets | No — most of these split timing evenly or via their own TTS engine's segment output, not designed for an externally-recorded single voiceover | Local (TTS/images can be swapped for free models) | Yes, decent | Closest existing category to what you want, but none of the ones I found treat "your voiceover + your pre-numbered image folder" as the two inputs to sync — they assume they're generating the TTS and images themselves, so they already know each segment's timing. You'd have to fork and rewrite the alignment logic anyway. |
| **auto-editor / ffmpeg CLI scripting by hand** | Yes, fully free | Yes if you write the script | Only if you manually build the alignment step | Local | Yes, if you write filters | This is essentially "build it yourself," just without the speech-alignment piece solved for you. |

**Bottom line:** nothing off-the-shelf does "single continuous voiceover + pre-numbered image folder → per-scene-exact synced video with effects and karaoke captions" out of the box. The closest category (open-source faceless-video generators) all assume *they* control the TTS and therefore already know segment boundaries — your case is harder because the voiceover is an external black box that has to be measured.

## The build-it-yourself path (what I built for you below)

Because there's a genuine gap, I built a working local pipeline rather than just describing one. It uses only free, local, open-source components — nothing that charges per video or per API call.

**Step by step, what it does:**

1. **Parse the script into scenes.** If you mark scenes explicitly (e.g. `Scene 1:`, `[Scene 2]`, or numbered lines), it uses those. Otherwise it splits on sentence boundaries — sentence 1 = scene 1, etc., exactly as you described.
2. **Transcribe the voiceover locally** with `faster-whisper` (CPU-friendly, no GPU or internet required after the model downloads once, no API key, no per-call cost) to get every spoken word with a start/end timestamp.
3. **Align script text to the transcript.** Since the narrator is reading your exact script, the word *order* is guaranteed to match — I use Python's built-in sequence-matching (same idea as a "diff") to anchor script words to transcribed words even where speech recognition made small errors, then compute the exact timestamp where each scene's narration starts and ends.
4. **Render each scene as its own clip** with `ffmpeg`, showing `<scene_number>.png/jpg` for exactly that scene's measured duration (accurate to a single video frame — as strict as digital video allows), with a rotating set of Ken Burns-style pan/zoom effects so scenes don't feel identical.
5. **Burn in word-by-word karaoke captions**, synced to the same word timestamps from step 2/3, using the ASS subtitle format (the same format CapCut/Premiere-style karaoke captions use).
6. **Concatenate all scene clips and mux in your original voiceover audio** as the single continuous soundtrack — no re-encoding drift, no re-syncing needed, since total clip length is built to equal total audio length exactly.
7. **A local Gradio dashboard** (`app.py`) gives you a browser-based UI on your own machine: upload script + voiceover + image folder (or a zip of images), pick effect style, click render, download the finished MP4.

All of this runs **100% locally** — no OpenAI/ElevenLabs/cloud video API bills, only your own CPU (or GPU, if available, for faster transcription).

## Packaging for "give it to someone with no setup"

You asked for a one-click desktop app. The full honest picture:

- **`faster-whisper`** has no heavy GPU-only dependency (unlike some Whisper variants that require PyTorch + CUDA) — this keeps the packaged app smaller and CPU-only machines fully supported.
- **`imageio-ffmpeg`** bundles a static ffmpeg binary automatically via pip — so the end user never installs ffmpeg separately.
- **PyInstaller** bundles Python itself, the whisper model loader, and ffmpeg into a single `.exe` (Windows) or `.app` (Mac) — the recipient double-clicks and it just works, no Python install required.
- **Trade-off to be transparent about:** the packaged app will be large (roughly 300-500MB, dominated by the speech-recognition model files) and PyInstaller builds are **platform-specific** — a Windows `.exe` must be built on Windows, a Mac `.app` on Mac. I've included the exact build script and instructions in `PACKAGING.md`; you (or I, if you hand me a Windows/Mac environment) run it once per target OS.

## Recommendation

Build it yourself using the pipeline below rather than adopting an existing tool. Nothing free matches your exact two inputs (externally-recorded single voiceover + your own pre-numbered image set) with frame-exact scene timing — every close match either charges per render (JSON2Video/Shotstack/Creatomate) or assumes it's generating the TTS itself (the open-source faceless-video repos), which breaks your timing requirement.

The pipeline I've built (`pipeline.py` + `app.py`) is a reasonably thin layer over two extremely well-proven free components — `faster-whisper` for alignment and `ffmpeg` for rendering — so it's not a fragile from-scratch research project, more an assembly of solved pieces. That's the fastest realistic path to something reliable, free, and packageable, and it's ready to run today; see `README.md` for setup and `PACKAGING.md` for turning it into a one-click app.

Sources:
- [WhisperX: word-level timestamps & diarization](https://github.com/m-bain/whisperx)
- [awesome-faceless: 80+ AI tools for faceless YouTube & Shorts](https://github.com/sasharun/awesome-faceless)
- [AI-Youtube-Shorts-Generator (Gemini + edge-tts + FFmpeg)](https://github.com/SaarD00/AI-Youtube-Shorts-Generator)
- [Viral-Faceless-Shorts-Generator](https://github.com/Dark2C/Viral-Faceless-Shorts-Generator)
- [PyInstaller + ffmpeg bundling gist](https://gist.github.com/Shreyaan/84458e718f41e51407a1effd2e93efdb)
- [PyInstaller ffmpeg discussion (Mac)](https://github.com/orgs/pyinstaller/discussions/8089)
