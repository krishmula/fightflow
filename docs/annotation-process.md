# Annotation Process: Combat Sports Technique Recognition

## Project context

Building a two-stage deep learning system to classify punch types (jab, cross, hook, uppercut) from video:

- **Stage 1**: Per-frame ResNet-18 baseline (single frame in → punch class out)
- **Stage 2**: CNN+LSTM clip-level classifier (sequence of frames in → punch class out)

Both stages need labeled training data. Since existing dataset annotations don't match our needs, we're creating our own from raw video.

## Core principle: annotate events, derive clips

The central design decision: **we do not cut clips during annotation**. Instead, we record a single frame index per punch — the moment of contact (or full arm extension) — into a CSV. Clips are generated later by a separate script that reads the CSV.

The rationale: annotation is expensive human labor; clip generation is cheap computation. By separating them, we get:

1. **Window size flexibility** — try 1s, 1.5s, 2s lead-ups by changing numbers in a config, no re-annotation needed
2. **Both stages from one annotation** — Stage 1 pulls `frames[contact]`; Stage 2 pulls `frames[contact - N_pre : contact + N_post]`
3. **Sampling rate flexibility** — subsample to T=16 or T=32 frames however we want
4. **Negative examples for free** — auto-sample frames ≥2s away from any labeled contact
5. **Ablation studies for the writeup** — compare multiple window sizes as an experimental result

Guiding rule: _annotate what's hard to change; derive what's easy to change._

## The annotation schema

A single CSV (or JSONL) file with one row per punch event:

```
video_id,     frame_index, punch_type, occluded, notes
fight_01.mp4, 1247,        jab,        false,    ""
fight_01.mp4, 1389,        cross,      false,    ""
fight_01.mp4, 1502,        hook,       true,     "partial occlusion"
fight_02.mp4, 203,         jab,        false,    ""
```

Minimum required columns: `video_id`, `frame_index`, `punch_type`. Optional but useful: `occluded`, `notes`, `fighter_id`, `camera_angle`.

**Frame index, not timestamp**: frame numbers are exact integers; timestamps drift with variable frame rate and fps differences across videos. Convert to time later if needed via `time = frame_index / fps`.

## The single most important decision: what does "contact frame" mean?

Before annotating anything, pick one definition and apply it consistently to every punch:

- Option A: moment of physical contact
- Option B: **moment of full arm extension (visual peak)** — recommended
- Option C: moment the fist starts moving forward

Full arm extension is recommended because it's the most visually identifiable and least often occluded. Inconsistency here poisons the training signal — the "moment" of a jab would effectively move around in the training data, and the model would struggle to learn a stable pattern.

## Tooling

Three realistic options:

**VIA (VGG Image Annotator)** — browser-based, zero install, supports temporal annotations with custom attribute dropdowns, exports CSV. Best for getting started immediately. Link: `https://www.robots.ox.ac.uk/~vgg/software/via/`

**Custom Jupyter notebook with ipywidgets** — ~50 lines of Python. Shows current frame, arrow keys to scrub, hotkeys `j/c/h/u` to label jab/cross/hook/uppercut, `n` for negative, logs to DataFrame, saves CSV periodically. Fastest per-event by a significant margin — worth the 30-minute setup cost for a multi-thousand-event project.

**CVAT** — more powerful but overkill for point events. Worth considering if we later add bounding boxes around the punching fist.

**DaVinci Resolve — do not use**. It's a video editor, not an annotation tool; reading timecodes and copy-pasting frame numbers into a spreadsheet is slow and error-prone at scale.

## The annotation loop

For each video:

1. Open in the annotation tool
2. Play at normal speed until a punch is imminent
3. Pause, scrub backwards frame-by-frame to find the defined contact frame
4. Hotkey the punch type → tool writes `video_id, frame_index, punch_type` to CSV
5. Flag with `occluded` or add `notes` if ambiguous
6. Resume playback after the punch, repeat

## Dataset scope

**Classes**: {jab, cross, hook, uppercut} + optionally a fifth "none/other" class for negatives.

**Target counts**: minimum ~200/class to get past random, ~500/class for a real shot at learnable patterns, ideally 1,000+/class. For 4 classes, realistic target is 2,000–4,000 total annotations.

**Negative examples**: sample random frames at least 2s away from any labeled contact. Generated automatically by a script, not hand-annotated. Critical if Stage 3 (inference on full fight video) is a goal — without negatives, the model confidently misclassifies stance-shifting, clinches, and feints as one of the four punches.

## Quality control practices

- **Calibration day first**: annotate 20 punches on a single video before committing. Measure seconds per punch. Extrapolate to total time. Decide feasibility before hour 37.
- **Self-consistency check**: double-label ~5% of punches on a different day. Measure agreement with your own earlier labels — this is a ceiling on model accuracy and a sanity check on label stability.
- **Log ambiguous cases** in the `notes` column rather than forcing a label. Can be excluded or revisited later.
- **Native fps annotation**: annotate at the video's native frame rate; do any subsampling at clip-extraction time, not at annotation time.

## Train/test splitting

Split the CSV by `video_id` (or by fight/event), **not** randomly by row. Otherwise the same fighter in the same lighting in the same venue appears in both train and test, and evaluation numbers reflect fighter-identity leakage rather than technique learning.

## Augmentation caveat

Horizontal flipping — a standard image augmentation — is dangerous for this task. Flipping an orthodox jab (lead hand) produces a visual that looks like a southpaw cross. For fine-grained punch-type classification, naive flipping corrupts the label. Either skip horizontal flips, or flip _and_ swap the jab↔cross label simultaneously.

## Downstream: from CSV to training data

A separate Python script will:

1. Read each CSV row
2. Open the corresponding video with OpenCV or decord
3. Extract the window: `frames[contact - N_pre : contact + N_post]`
4. Uniformly subsample to T frames (e.g., T=16 or T=32)
5. Resize/normalize per ResNet-18 input expectations
6. Save as tensor files or stream directly into a PyTorch Dataset

Stage 1 uses the same script but extracts only `frames[contact]`. The parameters `N_pre`, `N_post`, `T` live in a config, not in the annotations — they can be changed and re-run at will.

## Day-1 sanity checklist

1. Pick one video
2. Set up the chosen annotation tool
3. Decide on the contact frame definition and write it down
4. Annotate 20 punches end-to-end
5. Write the extraction script, run it on those 20 rows, confirm the resulting clips look right
6. Measure time per annotation; multiply to estimate total effort
7. Only then commit to the full annotation push

This catches tool problems, definition ambiguity, and pipeline bugs while the cost of fixing them is minutes, not days.
