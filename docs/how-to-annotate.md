# How to Annotate Videos

This guide covers the exact mechanics of annotating fight videos for the FightFlow dataset, plus what to look for when labeling each punch class.

> **Prerequisites**: You need `punch-annotator.ipynb` open in Jupyter. If you haven't set up the environment yet, run the install cell in that notebook once (`pip install av ipywidgets ipyevents pandas Pillow numpy`).

---

## 1. The Core Rule: Annotate the Peak Frame

For every punch, you are labeling **one single frame** — the moment of **full arm extension** (the visual peak of the punch). This is not the start of the wind-up, and not the moment of impact (which is often occluded). It is the frame where the arm is most extended and the punch is most visually identifiable.

**Why this frame?**
- It's the most consistent visual moment across different fighters and camera angles
- It's less often blocked by gloves or bodies than impact
- It gives the model the clearest signal for classification

**Consistency is everything.** If you drift between "impact frame" and "extension frame" within the same video, the model receives contradictory labels for the same visual pattern. Pick one definition and stick to it for every single punch in the dataset.

---

## 2. The Annotation Mechanics

### 2.1. Setup (per video)

1. Open `notebooks/punch-annotator.ipynb` in Jupyter
2. In the **Configuration** cell, edit `VIDEO_PATH` to point to the video you are annotating:
   ```python
   VIDEO_PATH = "../data/downloaded-videos/V7.mp4"
   ```
3. `CSV_PATH` should stay the same (`../data/annotations.csv`) — it accumulates across all videos
4. Run the **Launch** cell

### 2.2. The Interface

After launching, you will see:
- A video frame display
- Navigation buttons (Left, Right, Shift+Left, Shift+Right)
- Label buttons: **straight (s)**, **hook (h)**, **uppercut (u)**, **none (n)**
- Action buttons: **occluded (o)**, **undo (z)**, **save (s)**
- A frame jump input box

### 2.3. Keyboard Shortcuts

Click the video image first to give it focus, then use:

| Key | Action |
|---|---|
| `Left` / `Right` | Step backward / forward 1 frame |
| `Shift + Left` / `Shift + Right` | Step backward / forward 10 frames |
| `s` | Label current frame as **straight** |
| `h` | Label current frame as **hook** |
| `u` | Label current frame as **uppercut** |
| `n` | Label current frame as **none** (negative / non-punch) |
| `o` | Toggle "occluded" flag for the next label |
| `z` | Undo last annotation for this video |
| `s` | Save CSV immediately |

> **Note:** The `s` key both labels a frame as "straight" and manually saves the CSV. When in doubt, hit `s` — saving is idempotent.

### 2.4. The Annotation Loop

For each punch in the video:

1. **Play / scrub** through the video at normal speed until a punch is imminent
2. **Pause** and use `Left` / `Right` to scrub frame-by-frame until you find the **full arm extension** frame
3. **Hit the class hotkey** (`s`, `h`, or `u`) — the current frame is logged to the CSV
4. If the punching arm is partially hidden behind a body, glove, or opponent, **hit `o` first**, then the class key. This marks the frame as occluded
5. **Resume playback** after the punch retracts
6. Repeat

The notebook auto-saves every 10 annotations. You can also force-save anytime with the `s` key.

### 2.5. Resuming Work

The annotator automatically resumes at the frame after your last annotation for that video. If you restart the notebook, just re-run the launch cell with the same `VIDEO_PATH` — it will pick up where you left off.

---

## 3. What to Look For by Class

### 3.1. Straight (`s`)

A straight punch travels directly forward from the guard position toward the target.

**Visual cues:**
- Arm extends in a roughly straight line from shoulder to target
- Fist rotates palm-down (for orthodox fighters) or palm-up (for southpaw) at extension
- Non-punching hand stays near the chin in guard
- Shoulder rises to protect the chin on the punching side
- Hips and lead foot rotate into the punch

**What separates it from a hook:**
- The elbow stays either directly behind the fist or only slightly flared
- The path is linear, not arcing across the body
- The torso faces more forward (not turned sideways)

**Common edge cases:**
- **Slapping straight / open-hand strike:** Still label as straight if the arm extension pattern is straight-line. Note it in your head but label the mechanics, not the rule-set.
- **Superman punch:** The arm motion is still straight — label as straight.
- **Flicking straight / range-finder:** If it's an intentional strike (not a feint), label it. If it's just a hand-wave to measure distance, skip it.

### 3.2. Hook (`h`)

A hook travels in a horizontal arc, typically targeting the side of the head or body.

**Visual cues:**
- Elbow bends to roughly 90° and stays at that angle through the arc
- Fist moves horizontally across the body (either inside-to-outside or outside-to-inside)
- Torso rotates significantly; you often see the back of the fighter at full extension
- Weight transfers to the lead leg (for lead hook) or pivots on the rear foot (for rear hook)
- The punch "wraps around" guard rather than going through it

**What separates it from a straight:**
- The elbow is clearly bent and leading the motion, not trailing behind a straight arm
- The fist path is curved, not linear
- The hips rotate more than they drive forward

**Common edge cases:**
- **Looping overhand:** This is a hybrid — it has arc like a hook but travels on a vertical plane more like an uppercut. When in doubt, ask: is the fist moving horizontally at contact? If yes, hook. If the fist is coming downward, it's an overhand (which we don't have a class for). In that case, skip it or label as hook if it's closer to a hook arc.
- **Check hook:** A short hook thrown while moving backward. The range is short but the mechanics are pure hook — label as hook.
- **Shovel hook:** A hook to the body with a slight upward angle. Label as hook.

### 3.3. Uppercut (`u`)

An uppercut travels upward from below, typically targeting the chin or solar plexus.

**Visual cues:**
- Fist starts low (near hip or waist) and drives upward
- Elbow stays bent and tucked close to the body
- Torso dips slightly on the same side to load the punch, then drives upward
- The fist rotates palm-upward at extension (or palm-toward-body for some styles)
- Knee bends and extends to provide upward power

**What separates it from a hook:**
- The fist path is vertical / upward, not horizontal
- The elbow stays low and tight, not flared out at shoulder height
- The hips drive upward, not rotate sideways

**Common edge cases:**
- **Uppercut that becomes a hook at the end:** Some fighters loop an uppercut into a hook. Label based on the primary plane of motion in the 3-4 frames before extension. If it's mostly vertical → uppercut. If it's mostly horizontal → hook.
- **Body uppercut vs. shovel hook:** Body uppercuts come straight up the center; shovel hooks come around the side. If you can't tell, default to whichever motion is dominant in the wind-up.

### 3.4. None (`n`) — Negative Samples

Negative samples are frames where **no punch is occurring**.

**When to label `none`:**
- Fighter is in guard / neutral stance
- Fighter is moving, feinting, or shifting weight without committing to a strike
- Clinch / grappling (no striking motion)
- Defense: blocking, parrying, slipping, rolling
- Footwork only: stepping, pivoting, circling

**How to sample negatives:**
- Don't just grab random frames — make sure they are at least **2 seconds away** from any labeled punch frame
- Sample from different phases: early rounds, late rounds, between combinations, during clinches
- Try to balance negatives across different fighters and camera angles
- A good rule of thumb: for every 3 punches you label, label 1 negative frame

**Avoid:**
- Frames immediately before or after a punch (that's part of the punch sequence, not a true negative)
- Frames where the fighter is clearly winding up for a punch you haven't labeled yet
- Frames with severe motion blur where you can't actually tell what's happening (the model can't learn from these either)

---

## 4. The CSV Schema

Each row written to `data/annotations.csv` looks like this:

```
video_id,frame_index,punch_type,occluded,notes,timestamp
V7.mp4,15247,straight,False,,2026-04-28T10:15:32
```

| Column | Description |
|---|---|
| `video_id` | Filename of the video being annotated |
| `frame_index` | Exact frame number of the peak/extension frame |
| `punch_type` | One of: `straight`, `hook`, `uppercut`, `none` |
| `occluded` | `True` if the punching arm/hand is partially hidden |
| `notes` | Free text for edge cases (currently unused by notebook) |
| `timestamp` | ISO timestamp of when the label was created |

**Important:** We annotate by **frame index**, not timestamp. Frame indices are exact integers. Timestamps can drift with variable frame rates. If you need a timestamp later, compute it as `frame_index / fps`.

---

## 5. Quality Control Checklist

### Before you annotate a new video
- [ ] Confirm the video plays smoothly in the annotator
- [ ] Do a 5-punch calibration: label 5 punches, then scroll back and check they all show full arm extension. Adjust your eye if needed.

### While annotating
- [ ] Are you labeling the same visual moment every time? (Full arm extension)
- [ ] Are you flagging occlusions with `o`?
- [ ] Are your negative samples at least 2 seconds away from any punch?
- [ ] Are you saving periodically (or letting auto-save do its job)?

### After finishing a video
- [ ] Run the **Summary stats** cell in the notebook to verify counts per class
- [ ] Run `clip-validator.ipynb` to generate GIF previews and spot-check 10-20 random clips
- [ ] Look for drift: do the clips actually show the peak frame in the center?

### Self-consistency test (recommended every ~500 annotations)
- Pick a video you annotated a week ago
- Re-annotate 20 punches without looking at your old labels
- Compare: agreement should be >95%. If it's lower, your definition has drifted — stop and recalibrate.

---

## 6. Common Pitfalls

| Pitfall | Why it hurts | How to avoid |
|---|---|---|
| **Drifting contact definition** | Model sees the same visual pattern labeled as different classes | Pick "full arm extension" on day one and never deviate |
| **Annotating impact instead of extension** | Impact is often occluded; model learns inconsistent visual features | Always ask: "is the arm fully extended?" not "did it land?" |
| **Too many negatives near punches** | Model learns that "just before a punch" = none, hurting sequence models | Keep negatives ≥2s from any punch frame |
| **Ignoring occlusion** | Occluded punches look different; model may learn a corrupted template | Hit `o` before the label whenever the arm is partially hidden |
| **Rushing through combinations** | In a 5-punch combo, the 3rd and 4th punches are often sloppy or short | Still find the extension frame for each; don't guess |
| **Labeling feints** | Feints have no committed extension; labeling them poisons the class | If the arm doesn't fully extend, it's not a punch — skip it |
| **Forgetting to save** | Losing work is painful | Auto-save is every 10; hit `s` before switching videos |

---

## 7. Troubleshooting

**Scrubbing feels sluggish (backward steps take >300ms)**
- Your video probably has sparse keyframes. Re-encode with denser keyframes:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -g 10 -preset fast output.mp4
  ```

**Frame doesn't match what I see in VLC / QuickTime**
- Different players use different seek strategies. Trust the annotator — it decodes frame-by-frame from the nearest keyframe. If you need to verify, export a PNG from the annotator's current frame.

**Annotation isn't showing up in the CSV**
- Make sure you clicked the video image to give it keyboard focus
- Check that you didn't have the cursor in the Frame input box (that steals focus)
- Run the save cell or hit `s`

**I labeled something wrong 50 frames ago**
- The `z` (undo) button only undoes the most recent annotation for this video. For older mistakes, you can edit `annotations.csv` directly in a text editor or spreadsheet — just be careful not to break the format.

---

## 8. Quick Reference Card

Print this and keep it next to you while annotating:

```
CONTACT FRAME = FULL ARM EXTENSION (not impact)

s = straight   (linear, elbow behind fist, hips drive forward)
h = hook       (horizontal arc, elbow bent 90°, torso rotates)
u = uppercut   (vertical upward, elbow tight, hips drive up)
n = none       (neutral, defense, footwork, clinch — ≥2s from punches)
o = occluded   (hit BEFORE the class key if arm is hidden)
z = undo       (removes last annotation for this video)
```

---

## 9. From Annotations to Training Data

You don't need to cut clips yourself. A separate pipeline (`clip-validator.ipynb` and the dataset loaders) reads `annotations.csv` and:

1. Extracts a window around each contact frame (e.g., 1s before, 0.5s after)
2. Uniformly subsamples to T frames (e.g., 16 or 32)
3. Resizes and normalizes for ResNet-18 input
4. Saves as training tensors

This means:
- **Window size is flexible** — change it in config, no re-annotation needed
- **Sampling rate is flexible** — subsample however you want at extraction time
- **Negative examples can be auto-generated** — sample frames ≥2s from any labeled contact

Your job is just to label the exact peak frame. The pipeline handles the rest.
