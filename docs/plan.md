# MMA / Boxing Technique Analysis Project Plan

## 1. Project Goal

Build a system that can:

1. **Classify boxing techniques from short labeled clips** using a **CNN+LSTM** model.
2. **Process longer fight videos** using sliding-window inference to detect techniques astraight time.
3. **Feed structured detections into a pretrained VLM** to generate natural-language commentary and fight-level analysis.
4. Optionally, later explore **3D CNNs** or stronger video backbones as upgrades.

---

## 2. End-State Vision

### Minimum acceptable final system
```text
Input video
  -> CNN+LSTM technique recognizer
  -> detected techniques over time
  -> structured event summary
  -> pretrained VLM
  -> commentary / analysis
```

### Stronger Long Term Vision

```text
Input full fight video
  -> preprocessing + window generation
  -> technique detector (CNN+LSTM first, maybe 3D CNN later)
  -> timeline of events + confidence + timestamps
  -> optional pose / motion features
  -> pretrained VLM
  -> fight-wide commentary, tactical summaries, coaching-style feedback
```

### Success Criteria

## Have to Have

Clean dataset pipeline with labeled clips
Train / val / test split with minimal leakage
Per-frame CNN baseline trained and evaluated
CNN+LSTM trained and evaluated
CNN+LSTM compared against baseline
Confusion matrix and macro F1 analysis
Full-video sliding-window inference working
Timeline of detected techniques produced
Pretrained VLM generates commentary from detections

## Nice to Have

Clip-length ablation (8 / 16 / 32 frames)
Keypoint / motion summaries
Demo video with overlays
One comparison against a more modern model (3D CNN / VideoMAE / SlowFast)
Small interface or notebook demo

### Core Research Questions

#Core question
Does explicit temporal modeling with CNN+LSTM improve fine-grained boxing technique classification over a frame-only CNN baseline?

#Extension question
Can a clip-trained temporal model be applied over longer fight videos to produce usable technique timelines?

#Multimodal question
Can structured technique detections, when paired with a pretrained VLM, generate grounded natural-language commentary?

#Stretch question
Do stronger video backbones such as 3D CNNs outperform CNN+LSTM on this task?


### Overall Strategy

## Build in layers

Layer 1: clean labeled clips
Layer 2: frame-based CNN baseline
Layer 3: CNN+LSTM temporal model
Layer 4: evaluation and debugging
Layer 5: full-video sliding-window inference
Layer 6: structured event timeline
Layer 7: VLM commentary
Layer 8: stretch upgrades (3D CNNs, better VLM prompts, etc.)

### System Architecture

Labeled short clips
  -> frame CNN baseline
  -> CNN+LSTM temporal model
  -> best checkpoint
  -> sliding-window full-video inference
  -> event timeline (timestamps + labels + confidence)
  -> pretrained VLM
  -> natural-language fight analysis

### Workstreams

## 7. Workstreams

### Workstream A — Problem Definition

**Goals**
- Freeze task scope
- Define classes
- Define evaluation targets

**Decisions**
- Start with a small class set:
  - straight
  - straight
  - hook
  - uppercut
- Phase 1 = clip classification
- Phase 2 = full-video detection
- Phase 3 = VLM-based analysis

**Task definitions**
- **Phase 1 task:** Given a short clip containing one boxing technique, classify the technique.
- **Phase 2 task:** Given a longer fight video, detect techniques astraight time using sliding-window inference.
- **Phase 3 task:** Given the detected sequence of techniques and selected visual context, use a pretrained VLM to generate commentary and analysis.

### Workstream B — Data

**Goals**
Build a clean, trustworthy dataset pipeline from the local data we already have.

#### What We Actually Have

The `data/` directory contains three sub-directories:

```
data/
  Annotation_files/    # V1.xlsx … V10.xlsx  — frame-level punch labels
  RGB_videos/          # Meta_data.ods        — YouTube source links (no local video files)
  Skeleton_data/       # V1.npy  … V10.npy   — pre-extracted pose keypoints
```

**10 videos (V1–V10)** sourced from YouTube boxing matches. Only metadata (links) are stored locally; video files must be downloaded separately.

**Annotation format** (`.xlsx`, one file per video):
- 3 columns: `start`, `end`, `class`
- `start` / `end` are **frame numbers** (not seconds)
- Each row = one labeled punch event
- ~5,450 total labeled events astraight all videos

**Annotation class distribution (raw labels):**

| Class | Approx. Count |
|---|---|
| Straight | most common |
| Straight | common |
| Lead Hook | moderate |
| Rear Hook | moderate |
| Lead Uppercut | less common |
| Rear Uppercut | less common |

> **Note:** There are **6 punch classes** (not 4). Each is split into lead/rear variants. We can start with 4 merged classes (Straight, Straight, Hook, Uppercut) or train on all 6.

**Skeleton data format** (`.npy`, one file per video):
- Shape for most files: `(N_events, 25, 17, 2)`
  - `N_events` = number of annotated events in this video
  - `25` = frames per clip (fixed-length clip)
  - `17` = COCO-style body keypoints per person
  - `2` = (x, y) normalized pixel coordinates
- V6 has shape `(46497, 1, 17, 3)` — likely frame-level data with (x, y, confidence)
- Zero-padded entries exist where keypoints were not detected

This means pose keypoints are **already extracted and aligned** to annotation events. They can be used directly as an alternative or supplement to raw RGB frames.

---

#### Data Tasks

*B1. Video download*
- Use `yt-dlp` to download V1–V10 from the YouTube links in `RGB_videos/Meta_data.ods`
- Verify frame counts match annotation frame ranges
- Store locally in `data/RGB_videos/`

*B2. Annotation parsing*
For each `V{i}.xlsx`:
- Load the three columns: `start`, `end`, `class`
- Convert to a unified manifest CSV with columns:
  `video_id, start_frame, end_frame, class_raw, class_id, clip_duration_frames`

*B3. Label mapping*

Raw label → unified label. Two possible strategies:

**Strategy A — 4 merged classes (simpler):**
```json
{
  "Straight":            0,
  "Straight":          1,
  "Lead Hook":      2,
  "Rear Hook":      2,
  "Lead Uppercut":  3,
  "Rear Uppercut":  3
}
```

**Strategy B — 6 classes (harder, more informative):**
```json
{
  "Straight":            0,
  "Straight":          1,
  "Lead Hook":      2,
  "Rear Hook":      3,
  "Lead Uppercut":  4,
  "Rear Uppercut":  5
}
```

Start with Strategy A. Move to B only if class counts are sufficient.

*B4. Clip extraction from RGB video*
For each annotation event:
- Extract frames `[start_frame, end_frame]` from the downloaded video
- Center-pad or sample to a fixed clip length (16 or 32 frames)
- Save as individual clip directories or precompute frame tensors
- Also note: the `.npy` skeleton files give ~25 frames per event already

*B5. Pre-processing pipeline*

See **Workstream B-Pre** below for the full pre-processing plan.

*B6. Data cleaning*
Filter out events where:
- `end_frame - start_frame < 5` — too short, likely noise
- Skeleton data is all-zeros for > 50% of frames — keypoint detection failed
- Annotated duration > 60 frames — likely annotation error or overlap
- Overlapping annotations on the same video within 3 frames — deduplicate

*B7. Split design*
Split **by video ID** to prevent leakage (all events from one video stay in one split):

| Set | Videos |
|---|---|
| Train | V1, V2, V3, V4, V5, V6 |
| Val | V7, V8 |
| Test | V9, V10 |

This ensures no frame-level overlap between splits.

*B8. Class balancing*
Inspect per-class counts after cleaning. Options:
- **Weighted straight-entropy** — simplest, recommended first
- **Capped sampling** — cap majority class at 2× minority
- **Oversampling minority** — augment rare classes with flips/jitter

*B9. Storage format*
```
data/processed/
  dataset_manifest.csv   # all events with metadata
  label_map.json         # raw label → class id
  train.csv              # events in train split
  val.csv                # events in val split
  test.csv               # events in test split
  clips/                 # extracted RGB frame clips (optional)
  skeletons/             # aligned .npy skeleton clips (optional)
```

**Deliverables**
- `dataset_manifest.csv`
- `label_map.json`
- `train.csv`, `val.csv`, `test.csv`
- `scripts/download_videos.py`
- `scripts/extract_clips.py`
- `scripts/build_manifest.py`
- data quality notebook (class counts, duration distribution, skeleton check)

---

### Workstream B-Pre — Data Pre-processing

**Goals**
Transform raw annotations + video/skeleton files into model-ready tensors.

This project has **two modality tracks** that can be developed in parallel:
1. **RGB track** — use raw video frames (requires video download)
2. **Skeleton track** — use pre-extracted `.npy` keypoint arrays (available immediately)

Start with the **skeleton track** since the data is ready. Add the RGB track once videos are downloaded.

---

#### Pre-Pre: Dataset Audit (do this first)

Before any pre-processing, run an audit script to understand the data:
- Count events per class per video
- Check min/max/mean clip durations (in frames)
- Count zero-filled skeleton frames (detection failures)
- Verify skeleton `N_events` matches annotation row count per video
- Visualize 5 random samples per class (skeleton keypoint overlay)

---

#### Track 1: Skeleton Pre-processing

**Input:** `Skeleton_data/V{i}.npy` — shape `(N, 25, 17, 2)`

**Step 1 — Alignment check**
Verify that skeleton array index `k` corresponds exactly to annotation row `k` in `V{i}.xlsx`. If not, re-align using frame ranges.

**Step 2 — Clip length unification**
Most clips have 25 frames. Standardize to a fixed length T (e.g. 16 or 32):
- If clip has > T frames: sample T frames uniformly
- If clip has < T frames: repeat-pad the last frame

**Step 3 — Keypoint normalization**
Coordinates are already normalized (0–1 range). Additionally:
- Center the skeleton on the torso midpoint (hip center) to make it translation-invariant
- Optionally scale by person bounding-box height for scale-invariance

**Step 4 — Zero-mask detection**
For frames where all keypoints are zero, mark as invalid. Apply forward-fill from the last valid frame.

**Step 5 — Flatten / reshape for model**
Final tensor shape per clip: `(T, 17*2)` = `(T, 34)` flattened keypoints
Or keep as `(T, 17, 2)` for graph-based models.

**Step 6 — Augmentation (train only)**
- Horizontal flip (mirror left/right keypoints)
- Small Gaussian noise on keypoint positions
- Temporal jitter: randomly shift the clip window ±2 frames

---

#### Track 2: RGB Frame Pre-processing

**Input:** Downloaded video file + `(start_frame, end_frame)` from annotations

**Step 1 — Frame extraction**
Use `ffmpeg` or `cv2.VideoCapture` to extract frames `[start, end]`.
Target: 25 fps video, so 16 frames ≈ ~0.64s per clip.

**Step 2 — Clip sampling**
Sample T frames uniformly from the `[start, end]` range:
```python
indices = np.linspace(start, end, T, dtype=int)
```
This handles variable-duration clips cleanly.

**Step 3 — Spatial pre-processing per frame**
- Resize to `256×256`, then center-crop to `224×224`
- Normalize with ImageNet mean/std:
  ```python
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
  ```

**Step 4 — Augmentation (train only)**
- Random horizontal flip (whole clip, same decision per frame)
- Random crop `224×224` from `256×256` instead of center crop
- Color jitter: brightness ±0.2, contrast ±0.2 (mild)
- Temporal jitter: sample window shifted by ±5% of clip length

**Step 5 — Stack into tensor**
Final tensor per clip: `(T, 3, 224, 224)` — time × channels × H × W

**Step 6 — Optional: precompute CNN features**
If training the LSTM with a frozen CNN backbone, pre-extract features once:
```python
# shape: (N_clips, T, feature_dim)  e.g. (5000, 16, 512)
```
Save as a large `.npy` or HDF5 file. This avoids repeated forward passes through ResNet during training.

---

#### Preprocessing Output Summary

| Track | Input | Output tensor shape | Notes |
|---|---|---|---|
| Skeleton | `.npy` (N, 25, 17, 2) | `(T, 34)` per clip | Ready now |
| RGB clip | Raw video | `(T, 3, 224, 224)` | Needs download |
| CNN features | RGB clip | `(T, feature_dim)` | Optional precompute |

---

**Deliverables**
- `dataset_manifest.csv`
- `label_map.json`
- `train.csv`, `val.csv`, `test.csv`
- `scripts/download_videos.py`
- `scripts/extract_clips.py`
- `scripts/build_manifest.py`
- `data/processed/` directory with clean splits
- data quality / audit notebook

### Workstream C — Modeling

**Goals**
Train the baseline and the main temporal model.

**C1. Baseline model: frame CNN**

*Model*
- ResNet-18 baseline
- classify single frames or a small sampled set of frames
- aggregate frame predictions to clip prediction

*Questions*
- use center frame only?
- use N sampled frames and average?
- freeze backbone or fine-tune?

*Deliverables*
- baseline dataset loader
- baseline training script
- baseline metrics
- confusion matrix
- saved checkpoint

**C2. Main model: CNN+LSTM**

*Architecture*
```text
Frames in clip
  -> CNN encoder per frame
  -> feature vector per frame
  -> LSTM over sequence
  -> classifier head
  -> technique label
```

*Recommended starting setup*
- pretrained ResNet-18
- 16-frame clips
- 1-layer LSTM
- hidden size 256 or 512
- dropout in classifier head
- straight-entropy loss

*Design choices to test*
- clip length: 8 / 16 / 32 frames
- frame sampling rate
- frozen vs fine-tuned CNN
- hidden size
- 1-layer vs 2-layer LSTM
- bidirectional vs unidirectional
- dropout level

*Deliverables*
- clip loader
- CNN+LSTM training script
- best checkpoint
- comparison table vs baseline
- confusion matrix
- ablation results

**C3. Training setup**

*Training features*
- PyTorch training loop
- optimizer: Adam or AdamW
- LR scheduler
- early stopping
- checkpoint saving
- mixed precision if available
- gradient clipping if needed
- experiment logging

*Data preprocessing*
- resize / crop to 224x224
- ImageNet normalization
- optional horizontal flips
- random crops
- temporal jitter

*Evaluation metrics*
- accuracy
- macro F1
- per-class precision / recall / F1
- confusion matrix
- top-2 accuracy (optional)

### Workstream D — Full-Video Inference

**Goals**
Turn the clip classifier into a usable long-video detector.

**D1. Windowing strategy**
Split the full video into overlapping windows.

Example:
```text
0.0–2.0s
0.5–2.5s
1.0–3.0s
1.5–3.5s
...
```

**D2. Window prediction**
For each window:
- run CNN+LSTM
- output label + confidence

**D3. Post-processing**
Merge predictions using:
- confidence thresholding
- smoothing
- merging adjacent same-label windows
- non-max suppression or similar logic
- removal of very short spurious detections

**D4. Event timeline output**
Generate:
```csv
start_time,end_time,label,confidence
12.1,12.6,straight,0.91
14.8,15.5,straight,0.87
27.4,27.9,hook,0.79
```

**D5. Optional visualization**
- overlay label and confidence on video
- produce annotated demo clip
- plot event timeline

**Deliverables**
- long-video inference script
- event merging logic
- timeline JSON / CSV
- optional annotated video

### Workstream E — VLM Integration

**Goals**
Generate grounded commentary from model outputs.

**Principle**
- The VLM is not the core detector.
- The VLM sits on top of the detector outputs.

**E1. Define the VLM’s job**
*Possible jobs:*
- describe a clip
- summarize detected techniques astraight a video
- explain visible patterns
- generate commentary
- answer fight-related questions

*Recommended first job:*
Given:
- event timeline
- selected frames or clips
- confidence values

Generate:
- short commentary
- pattern summary
- tactical observations

**E2. Inputs to the VLM**
*Minimum*
- timestamps
- labels
- confidences

*Better*
- timestamps
- labels
- confidences
- selected frames
- short video segments
- optional motion summaries

*Example input schema*
```json
{
  "video_id": "fight_01_round_1",
  "events": [
    {"start": 12.1, "end": 12.6, "label": "straight", "confidence": 0.91},
    {"start": 14.8, "end": 15.5, "label": "straight", "confidence": 0.87},
    {"start": 27.4, "end": 27.9, "label": "hook", "confidence": 0.79}
  ]
}
```

**E3. Prompt design**
Prompts should keep the VLM grounded.

*Example prompt*
```text
You are analyzing a boxing video.

Detected events:
- 12.1–12.6s: straight (0.91)
- 14.8–15.5s: straight (0.87)
- 27.4–27.9s: hook (0.79)

Use the visual context and these detections to describe:
1. what techniques are being used,
2. whether there are repeated combinations or patterns,
3. a short commentary of the exchange.

Do not invent techniques not supported by the detections or visible evidence.
```

**E4. VLM experiment levels**
- **Level 1:** Structured detections only
- **Level 2:** Structured detections + selected frames
- **Level 3:** Structured detections + selected clips / broader video context

**E5. VLM evaluation**
Since formal benchmarking may be hard, evaluate:
- coherence
- groundedness
- hallucination rate
- usefulness
- agreement with detection timeline

**Deliverables**
- VLM input formatter
- prompt templates
- sample outputs
- hallucination / groundedness notes
- one end-to-end demo

### Workstream F — Stretch Upgrades

**F1. 3D CNN exploration**
Possible candidates:
- C3D
- I3D
- SlowFast
- X3D

Goal:
- compare against CNN+LSTM
- see whether joint spatiotemporal modeling improves results

**F2. Transformer / self-supervised models**
Possible candidates:
- VideoMAE
- MViT
- TimeSformer

Goal:
- test stronger pretrained video representations
- compare against CNN+LSTM

**F3. Pose / keypoint branch**
Possible features:
- hand velocity
- punch direction
- torso rotation
- stance changes
- guard return timing

**F4. Better VLM usage**
- round-by-round summary
- fight-level tactical summary
- fighter comparison
- Q&A over video
- retrieval of specific exchanges

---

## 8. Project Phases

### Phase 1 — Foundation
**Goal:** Make the dataset usable and the task precise.

**Tasks**
- [ ] Decide: 4 merged classes vs. 6 fine-grained classes
- [ ] Run dataset audit script (class counts, duration distribution, skeleton quality)
- [ ] Download videos V1–V10 via `yt-dlp`
- [ ] Parse all `.xlsx` annotation files into `dataset_manifest.csv`
- [ ] Build `label_map.json`
- [ ] Clean events (filter very short / long durations, bad skeletons)
- [ ] Split by video ID: train (V1–V6), val (V7–V8), test (V9–V10)
- [ ] Run skeleton pre-processing pipeline (normalize, pad, center)
- [ ] Extract RGB clips from downloaded videos
- [ ] Visualize 5 random samples per class (skeleton overlay or raw frames)
- [ ] Verify skeleton `N_events` matches annotation row count per video

**Exit criteria**
- `dataset_manifest.csv`, `train.csv`, `val.csv`, `test.csv` exist
- Skeleton tensors load correctly with shape `(T, 34)` per clip
- RGB clips load correctly
- Class counts per split are known and recorded
- No data leakage between splits (verified by video ID)

### Phase 2 — Baseline
**Goal:** Train a frame-only CNN baseline.

**Tasks**
- [ ] Build frame dataset loader
- [ ] Implement ResNet-18 baseline
- [ ] Choose frame aggregation method
- [ ] Train baseline
- [ ] Evaluate baseline
- [ ] Inspect errors

**Exit criteria**
- baseline checkpoint saved
- accuracy and macro F1 recorded
- confusion matrix created

### Phase 3 — CNN+LSTM Core
**Goal:** Train the must-have temporal model.

**Tasks**
- [ ] Implement clip loader
- [ ] Extract frame features
- [ ] Implement LSTM head
- [ ] Train first working model
- [ ] Debug data ordering / shapes
- [ ] Tune hyperparameters
- [ ] Compare against baseline

**Exit criteria**
- stable training
- best checkpoint saved
- results table produced
- confusion matrix comparison produced

### Phase 4 — Controlled Experiments
**Goal:** Understand the model, not just train it.

**Tasks**
- [ ] clip-length ablation
- [ ] frozen vs fine-tuned CNN
- [ ] class imbalance experiments
- [ ] augmentation experiments
- [ ] hard-confusion analysis

**Exit criteria**
- ablation table
- final model selection
- final chosen checkpoint

### Phase 5 — Full-Video Inference
**Goal:** Run the model over longer videos.

**Tasks**
- [ ] implement sliding-window inference
- [ ] run on a long fight / round
- [ ] merge overlapping detections
- [ ] tune thresholds
- [ ] produce event timeline
- [ ] optionally create annotated demo video

**Exit criteria**
- one long video processed end-to-end
- event timeline looks reasonable
- demo artifact exists

### Phase 6 — VLM Integration
**Goal:** Generate natural-language commentary.

**Tasks**
- [ ] define VLM input / output format
- [ ] create prompt templates
- [ ] feed timeline into VLM
- [ ] optionally include frames / clips
- [ ] compare prompt variants
- [ ] evaluate groundedness

**Exit criteria**
- one end-to-end demo:
  - input video
  - technique detections
  - generated commentary

### Phase 7 — Upgrade Phase
**Goal:** Explore stronger models and richer analysis.

**Options**
- [ ] 3D CNN comparison
- [ ] transformer-based video model
- [ ] keypoint features
- [ ] better VLM prompting
- [ ] tactical summaries
- [ ] fighter-specific analysis
- [ ] search / retrieval interface

**Exit criteria**
Any one of:
- CNN+LSTM vs 3D CNN comparison
- stronger VLM analysis layer
- more robust full-video system
- enriched feature pipeline

---

## 9. Full Task Backlog

### A. Data Checklist

**Audit**
- [ ] run dataset audit notebook (class counts, clip durations, skeleton zero-fill rate)
- [ ] verify skeleton `N_events` matches annotation row count for each video
- [ ] visualize 5 random events per class (skeleton or RGB)

**Annotations**
- [ ] parse V1–V10 `.xlsx` files into unified `dataset_manifest.csv`
- [ ] decide: 4 merged classes vs. 6 fine-grained classes
- [ ] write and save `label_map.json`
- [ ] filter events: too short (< 5 frames), too long (> 60 frames), bad skeleton

**Video download**
- [ ] install `yt-dlp`
- [ ] download V1–V10 from YouTube links in `RGB_videos/Meta_data.ods`
- [ ] verify downloaded video frame counts align with annotation frame numbers
- [ ] store in `data/RGB_videos/`

**Skeleton pre-processing**
- [ ] align skeleton arrays to annotation rows
- [ ] standardize clip length to T frames (pad / sample)
- [ ] normalize keypoints (center on torso, optionally scale by height)
- [ ] forward-fill zero keypoint frames
- [ ] save processed skeletons to `data/processed/skeletons/`

**RGB pre-processing**
- [ ] write clip extraction script using frame ranges from manifest
- [ ] resize to 256×256, center-crop to 224×224
- [ ] apply ImageNet normalization
- [ ] implement augmentation: flip, random crop, color jitter, temporal jitter
- [ ] optionally precompute CNN features and save as `.npy`

**Splits**
- [ ] split by video ID: train (V1–V6), val (V7–V8), test (V9–V10)
- [ ] verify no video overlap between splits
- [ ] count clips per class per split
- [ ] inspect class imbalance and decide on balancing strategy
- [ ] finalize `train.csv`, `val.csv`, `test.csv`

### B. Infrastructure Checklist
- [ ] choose framework (PyTorch)
- [ ] define repository structure
- [ ] implement config system
- [ ] implement logging
- [ ] implement checkpoint saving
- [ ] set seeds for reproducibility
- [ ] create experiment naming convention
- [ ] create metric utilities
- [ ] create plotting utilities
- [ ] create inference utilities

### C. Baseline Checklist
- [ ] implement frame dataset loader
- [ ] implement ResNet-18 baseline
- [ ] choose frame aggregation method
- [ ] train first baseline
- [ ] tune learning rate
- [ ] compute accuracy / macro F1
- [ ] create confusion matrix
- [ ] inspect failed examples
- [ ] save final baseline checkpoint

### D. CNN+LSTM Checklist
- [ ] implement clip loader
- [ ] implement sequence sampling
- [ ] extract frame features with CNN
- [ ] implement LSTM head
- [ ] implement classifier head
- [ ] train first run
- [ ] debug shape / ordering issues
- [ ] tune hidden size
- [ ] tune clip length
- [ ] test frozen vs fine-tuned CNN
- [ ] compare to baseline
- [ ] save best checkpoint
- [ ] generate confusion matrices
- [ ] document final settings

### E. Analysis Checklist
- [ ] results table
- [ ] per-class precision / recall / F1
- [ ] macro F1
- [ ] accuracy
- [ ] confusion matrix
- [ ] misclassification gallery
- [ ] clip-length ablation
- [ ] interpret main failure modes

### F. Full-Video Inference Checklist
- [ ] design sliding-window policy
- [ ] implement full-video loader
- [ ] run predictions astraight long video
- [ ] smooth predictions
- [ ] merge duplicate detections
- [ ] export timeline JSON / CSV
- [ ] inspect event quality manually
- [ ] create timeline visualization
- [ ] optionally render annotated video

### G. VLM Checklist
- [ ] choose pretrained VLM
- [ ] define prompt template
- [ ] define event JSON schema
- [ ] select video context to send
- [ ] test structured-input-only prompting
- [ ] test structured-input-plus-frames prompting
- [ ] compare prompt variants
- [ ] review hallucinations
- [ ] refine groundedness instructions
- [ ] save example demos

### H. Stretch Checklist
- [ ] pick 3D CNN candidate
- [ ] adapt dataset loader
- [ ] fine-tune modern model
- [ ] compare to CNN+LSTM
- [ ] try keypoint features
- [ ] try fighter-specific summaries
- [ ] explore retrieval / search demo

---

## 10. Priority Stack

**Tier 1 — Mandatory**
1. dataset audit + manifest build
2. skeleton pre-processing pipeline
3. RGB video download + clip extraction
4. frame-based CNN baseline (RGB track)
5. CNN+LSTM temporal model
6. proper evaluation
7. full-video sliding-window inference
8. VLM commentary layer

**Tier 2 — Strong improvements**
1. clip-length ablation
2. error analysis
3. better post-processing
4. better VLM prompts
5. motion summaries

**Tier 3 — Stretch**
1. 3D CNN
2. VideoMAE / transformer
3. advanced VLM usage
4. richer coaching analysis

---

## 11. Risks and Mitigations

### Risk 1 — Noisy labels
**Problem:** Punch labels may be ambiguous.
**Mitigation:**
- restrict to 4 clean classes first
- manually inspect samples
- drop ambiguous cases

### Risk 2 — Too little data
**Problem:** Not enough clips per class.
**Mitigation:**
- use pretrained CNN
- use augmentation
- use weighted loss
- reduce number of classes if needed

### Risk 3 — Data leakage
**Problem:** Train/test clips from the same fight may leak context.
**Mitigation:**
- split by fight or boxer when possible
- track metadata carefully

### Risk 4 — CNN+LSTM does not improve much
**Problem:** Temporal model may not outperform baseline significantly.
**Mitigation:**
- tune clip lengths
- verify temporal span is meaningful
- inspect failure cases
- report honest result

### Risk 5 — Full-video inference is noisy
**Problem:** Clip classifier works, but long-video detections are unstable.
**Mitigation:**
- use overlapping windows
- threshold low-confidence predictions
- merge adjacent detections
- smooth sequence outputs

### Risk 6 — VLM hallucination
**Problem:** Commentary sounds plausible but unsupported.
**Mitigation:**
- ground prompt in structured detections
- provide confidence values
- instruct model not to invent unseen techniques
- optionally include relevant frames only

---

## 12. Deliverables

**Technical artifacts**
- dataset manifest
- extraction scripts
- training code
- evaluation notebook
- baseline checkpoint
- CNN+LSTM checkpoint
- long-video inference script
- timeline JSON / CSV
- VLM prompting script

**Visual artifacts**
- class distribution chart
- training curves
- confusion matrices
- timeline visualization
- annotated demo video
- sample commentary outputs

**Written artifacts**
- architecture diagram
- methodology write-up
- results section
- error analysis
- future work section

---

## 13. Minimum Viable Implementation

If scope becomes tight, the narrowest acceptable version is:
1. choose 4 classes
2. extract clean labeled clips
3. train ResNet-18 baseline
4. train ResNet-18 + LSTM
5. compare metrics
6. run CNN+LSTM on one long fight video with sliding windows
7. produce timestamped technique timeline
8. feed timeline + selected frames to pretrained VLM
9. generate commentary

*This is enough to form a complete end-to-end system.*

---

## 14. Immediate Next Actions

1. **Run the dataset audit** *(do this today — no downloads needed)*
   - Load all `.xlsx` files, count events per class per video
   - Check skeleton `.npy` shapes and zero-fill rate
   - Decide: 4 merged classes vs. 6 fine-grained classes

2. **Build `dataset_manifest.csv`**
   - Parse V1–V10 annotations into one unified CSV
   - Columns: `video_id, start_frame, end_frame, class_raw, class_id, clip_duration_frames`
   - Save `label_map.json`

3. **Start the skeleton pipeline** *(fastest path to a working model)*
   - Skeleton data is already available — no video download needed
   - Normalize, pad/sample to fixed T=25 frames, center on torso
   - Flatten to `(T, 34)` tensors and save
   - Train a quick LSTM on skeleton features to validate the pipeline end-to-end

4. **Download videos with `yt-dlp`**
   - Use links in `RGB_videos/Meta_data.ods`
   - Verify frame alignment with annotations

5. **Build the RGB clip extraction script**
   - Extract `[start_frame, end_frame]` clips
   - Resize → normalize → stack to `(T, 3, 224, 224)`

6. **Train the CNN baseline**
   - ResNet-18 on single center frames (or average of T sampled frames)
   - Establish a sanity-check reference metric

7. **Build the CNN+LSTM**
   - This is the most important modeling milestone