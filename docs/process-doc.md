# Combat Sports Punch Recognition — Project Process Document

## Overview

This project builds a punch classification system from boxing/MMA heavy bag footage, capable of recognizing four punch types from video. The pipeline progresses through data collection, manual annotation, a series of CNN baselines, and a CNN+LSTM temporal model, with the longer-term goal of attaching natural language coaching feedback via an LLM API.

---

## 1. Classes

| Label      | Description                                                                                                            |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| `straight` | Jab and cross combined — both are mechanically identical in terms of arm extension, differentiated only by stance hand |
| `hook`     | Horizontal arc punch, elbow bent at roughly 90° at contact                                                             |
| `uppercut` | Vertical rising punch, targeting the chin                                                                              |
| `negative` | Non-punch states — guard position, standing still, footwork, transitions                                               |

> **Note on jab vs. cross:** Jab and cross are merged into a single `straight` class. This is intentional — it simplifies the classification problem and avoids label noise introduced by horizontal flip augmentation (a mirrored jab is visually identical to a cross).

---

## 2. Data Collection

### Source

YouTube videos — specifically heavy bag drills and tutorials. Search terms like _"boxing heavy bag drills"_, _"heavy bag tutorial"_, _"boxing combinations heavy bag"_ yield dense, well-lit, single-fighter footage, which is ideal.

### Resolution & FPS

- **Current phase:** 360p, 24–30 FPS
- **Future phase:** 1080p, same FPS range (slotted after initial pipeline is validated)

FPS variation across the 24–30 range is deliberately left unnormalized. The slight difference in temporal dynamics between a 24fps clip and a 30fps clip adds natural variance to the dataset, which improves model generalization — a 15-frame window at 24fps covers slightly more time than at 30fps, meaning the model sees a broader range of windup/retraction durations.

### On 360p vs. 1080p

The resolution difference matters less than it might seem, for one key reason: both ResNet-18 and VGG-16 resize all inputs to 224×224 before any computation. A 360p frame (640×360) and a 1080p frame (1920×1080) are both downsampled to the same thing, so the information advantage of 1080p is largely destroyed at that step.

What 1080p genuinely provides is more detail _before_ the resize — finer glove texture, cleaner arm edges, less compression artifact. Whether that survives the downscale in a way the model can use is unclear. For punch classification, the discriminative signal is **motion and body mechanics**, not fine texture detail, so the practical improvement is expected to be modest.

The more meaningful difference between 360p and 1080p footage is correlated quality — 1080p YouTube videos tend to be better lit, steadier camera, and cleaner backgrounds. This is a **data quality effect**, not a resolution effect. 1080p is better treated as a robustness test (does the model generalize across resolutions?) than an expected performance booster. Validate the full pipeline on 360p first.

### Video-Level Train/Val/Test Split

Splits are performed **at the video level**, not at the frame or clip level. This is critical — if clips from the same video appear in both train and test, the model will learn video-specific features (background, lighting, bag texture, fighter appearance) rather than punch mechanics, producing inflated and misleading metrics.

Suggested split: **70% train / 15% val / 15% test** by video count, stratified to preserve rough class balance across splits.

---

## 3. Annotation

### Tooling

A custom annotation tool built in-house, with key bindings for rapid frame-stepping and class labeling. This dramatically reduces the time cost of manual annotation compared to general-purpose tools.

### Annotation Protocol

**Contact point definition:** For each punch, the annotated frame is the frame of _maximum extension_ — the point at which the arm is straightest (for straights), or the elbow is at its tightest arc (for hooks), or the fist is at its highest point (for uppercuts). This is consistently the moment of or immediately before contact with the bag.

**Negative class:** Annotated at moments where the fighter is stationary, in a guard stance, or performing non-punch movement (footwork, head movement, stance shifts). The negative class should represent a clean, unambiguous non-punch state — avoid annotating frames mid-transition between punches, as these are ambiguous and will degrade the decision boundary.

### Output Format

A CSV file with one row per annotation:

```
video_id, frame_index, class
vid_001, 1042, straight
vid_001, 1187, hook
vid_002, 334, negative
...
```

### Annotation Consistency

If more than one annotator is working on this:

- Define "maximum extension" with visual examples for each class before starting
- Annotate a shared calibration set of ~50 clips and compare frame choices
- A 2–3 frame disagreement on the contact point shifts the entire 15-frame window, which matters for the LSTM

---

## 4. Clip Extraction (for CNN+LSTM)

For each annotated contact point frame `f`:

```
clip = [f - 10, f - 9, ..., f, f + 1, ..., f + 5]
```

- **10 frames before** the contact point — captures the windup and approach
- **Contact frame** — peak extension
- **5 frames after** — captures retraction

This gives a **16-frame clip** per annotation. This asymmetry (10 before, 5 after) is intentional — the windup carries more class-discriminative motion than the retraction. The total clip length of 16 frames is a fixed constant that should be defined once in the codebase and referenced everywhere.

### Clip Overlap / Bleeding

When two punches occur in quick succession (e.g., a jab-cross combo), the 15-frame windows will overlap. The retraction of punch A may appear in the windup of punch B's clip. This is a known dataset limitation. It is not fully avoidable given the annotation strategy, and the asymmetric window (heavier weight on pre-contact frames) partially mitigates it. Clips where the nearest neighboring annotation is fewer than ~12 frames away can optionally be excluded or flagged during dataset construction.

### Frame Extraction to Disk

All frames are extracted to disk upfront (not on-the-fly during training). This avoids dataloader bottlenecks during training and becomes especially important at 1080p. Directory structure suggestion:

```
frames/
  vid_001/
    0001.jpg
    0002.jpg
    ...
  vid_002/
    ...
```

Clips are then assembled at dataset load time by reading the relevant frame indices from the CSV.

---

## 5. Augmentation

Applied during training only — never on validation or test data.

| Augmentation                 | Details                                       |
| ---------------------------- | --------------------------------------------- |
| Small rotation               | ±10–15° — simulates camera angle variation    |
| Brightness / contrast jitter | Accounts for lighting variation across videos |
| Gaussian noise               | Low intensity — improves robustness           |
| Random crop / resize         | Minor spatial jitter                          |

### What NOT to do

**Do not apply horizontal flips (mirror image).** A mirrored jab is visually identical to a cross. Even though jab and cross are merged into `straight`, the habit of applying mirror augmentation should be avoided — if the class structure is ever refined to split jab and cross, mirror augmentation would introduce direct label noise. If a mirror flip is ever used deliberately, document the rationale explicitly.

---

## 6. Preprocessing

### ImageNet Normalization

ResNet-18 and VGG-16 are both ImageNet-pretrained. Input frames must be normalized with ImageNet statistics before being passed to either backbone:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Skipping this causes subtle but real training instability and degrades transfer learning quality.

### Frame Resize

Resize all frames to the input size expected by the backbone (224×224 for both ResNet-18 and VGG-16) before saving to disk, or as the first step in the dataloader transform pipeline.

---

## 7. Pose Estimation (MediaPipe)

Pose estimation is an additional input modality to be explored alongside the CNN+LSTM architecture. Instead of (or in addition to) raw pixel features, the model receives per-frame joint coordinates — wrist, elbow, shoulder, hip, etc. — which describe the _geometry_ of the punch rather than its appearance.

For punch classification, joint trajectory over 16 frames is highly discriminative: a straight shoots the wrist forward along a horizontal axis, a hook arcs the wrist laterally with a bent elbow, an uppercut drives the wrist upward. This is information that exists in the CNN features implicitly but is explicit in pose keypoints.

### Tooling

**MediaPipe Pose** — produces 33 keypoints × 3 coordinates (x, y, z) = 99-dim vector per frame. Runs as a preprocessing step, not inside the training loop.

### Known Failure Modes in Heavy Bag Footage

MediaPipe can struggle in this domain:

- The heavy bag partially occludes the body and arms
- Fighters are sometimes side-on to the camera
- Fast punches cause motion blur that throws off keypoint detection

Per-frame confidence scores should be inspected. Low-confidence keypoints should be handled explicitly — options include zeroing them out, interpolating from neighboring frames, or flagging and dropping clips where confidence falls below a threshold on too many frames.

### Preprocessing & Storage

Keypoints are extracted per frame and saved to disk alongside the frame images — one `.npy` file per clip:

```
poses/
  vid_001/
    clip_1042.npy   # shape: (16, 99)
    clip_1187.npy
    ...
```

At dataset load time, both the frame sequence and the keypoint sequence are loaded and used as inputs to the model.

### Architectural Options

**Option A — Pose-only LSTM (no CNN)**
Run MediaPipe per frame → 99-dim keypoint vector → sequence of 16 × 99 vectors → LSTM → 4-class softmax. No CNN at all. Lighter, faster, and fully interpretable — joint trajectories can be visualized directly to understand model decisions.

**Option B — CNN + Pose, concatenated (recommended)**
For each frame, extract both a CNN feature vector and a pose keypoint vector, then concatenate them before passing to the LSTM:

```
Frame → ResNet-18 (frozen) → 512-dim vector  ┐
Frame → MediaPipe           →  99-dim vector  ┘ → concat → 611-dim vector per frame
                                                              ↓
                                              Sequence of 16 × 611-dim vectors
                                                              ↓
                                                    Single-layer LSTM (hidden 256)
                                                              ↓
                                                   Linear(256, 4) → Softmax
```

The CNN contributes appearance — what the punch _looks_ like. The pose estimation contributes joint trajectory — the _geometry_ of how the arm moves. The LSTM learns temporal patterns over the combined representation. Neither alone is complete: pose without appearance misses glove position and body rotation; appearance without pose buries joint trajectory in pixel noise.

Everything else about the architecture stays the same — same LSTM, same final linear layer, same 4-class softmax. The only change is the input dimensionality at each timestep (611 instead of 512).

### Experimental Design

Pose estimation is slotted as a Stage 2 ablation, producing a clean three-way comparison:

| Model         | Input per timestep   | What it tests         |
| ------------- | -------------------- | --------------------- |
| CNN+LSTM      | 512-dim CNN features | Appearance only       |
| Pose+LSTM     | 99-dim keypoints     | Skeleton only         |
| CNN+Pose+LSTM | 611-dim concatenated | Appearance + skeleton |

This is a well-motivated academic result — it directly quantifies the contribution of each modality to punch classification performance.

---

## 8. Model Pipeline

### Stage 1a — Lightweight Custom CNN (Baseline)

A simple, few-layer CNN trained from scratch on single annotated frames. Purpose: establish a floor-level baseline and validate the data pipeline end-to-end before introducing pretrained weights.

### Stage 1b — ResNet-18 (Fine-tuned)

ImageNet-pretrained ResNet-18, fine-tuned on single annotated frames (no temporal context). Expected to perform reasonably well on hooks and uppercuts (spatially distinct) but struggle to distinguish punch classes that look similar at peak extension. This confusion matrix is a useful diagnostic — it motivates the shift to temporal modeling.

### Stage 1c — VGG-16 (Fine-tuned)

Same setup as ResNet-18 but with the deeper VGG-16 backbone. Comparison between 1b and 1c gives a sense of how much backbone capacity matters for the per-frame task.

### Stage 2 — CNN+LSTM

Architecture based on the LRCN-style approach described in [this paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10143200).

**Architecture:**

```
Input: 16 frames × (C × H × W)
  → ResNet-18 (frozen) → 512-dim feature vector per frame
  → Sequence of 16 × 512 feature vectors
  → Single-layer LSTM (hidden size 256)
  → Final hidden state → Linear(256, 4) → Softmax
```

**Key design decisions:**

- The CNN backbone (ResNet-18) is **frozen** during LSTM training. This keeps the feature extractor fixed, makes the Stage 1 vs. Stage 2 comparison clean and interpretable (same features, different sequence modeling), and reduces the risk of overfitting given dataset size.
- A single LSTM layer is used initially. Stacking LSTM layers can be explored as an ablation if Stage 2 results are underwhelming.
- Hidden size of 256 is a starting point — ablate against 128 and 512 if compute allows.

### Stage 3 (Stretch Goal) — LLM Coaching Feedback

Structured predictions from the CNN+LSTM (class + confidence per clip) are piped to an LLM API to generate natural language coaching feedback. This is **not** end-to-end trained — it is a post-hoc generation step layered on top of the classifier output. Implementation is explicitly deferred until Stages 1 and 2 are complete and producing reliable predictions.

---

## 9. Training Details

### Loss Function

Cross-entropy loss. If class imbalance is significant after annotation (straights tend to dominate in heavy bag footage), use **class-weighted cross-entropy** or consider **focal loss** to prevent the model from over-indexing on majority classes.

### Class Balance Strategy

Aim for a roughly balanced annotation distribution across the four classes. Track annotation counts per class throughout the annotation phase. If imbalance is unavoidable, apply oversampling of minority classes at the dataset level before resorting to loss weighting.

### Optimizer

Adam or AdamW. Start with lr=1e-4 for fine-tuning pretrained backbones; the LSTM head can use a slightly higher rate if training separately.

---

## 10. Evaluation

### Metrics

- **Per-class precision, recall, F1** — not just overall accuracy
- **Confusion matrix** — the primary diagnostic tool for fine-grained classification. Reveals whether failures are spatial (e.g., hook vs. uppercut confusion — a backbone problem) or temporal (e.g., straight vs. hook confusion — a sequence modeling problem)
- **Overall accuracy** — reported for comparability but not the primary signal

### Evaluation Protocol

All evaluation is on the held-out test set (video-level split). Validation set is used for hyperparameter tuning and early stopping only.

---

## 11. Ablations & Improvement Path

Once baseline results are in, iterate in the following directions:

1. **Clip length** — ablate 16-frame window against shorter (12) and longer (20) windows
2. **LSTM depth** — single layer vs. stacked (2-layer)
3. **LSTM hidden size** — 128 / 256 / 512
4. **Pose modality** — CNN+LSTM vs. Pose+LSTM vs. CNN+Pose+LSTM (three-way comparison)
5. **Unfreezing the CNN** — end-to-end fine-tuning as a late-stage experiment (higher overfitting risk, needs sufficient data)
6. **1080p footage** — higher resolution inputs as a robustness test once the 360p pipeline is validated

---

## 12. Timeline

| Days  | Phase                                                                    |
| ----- | ------------------------------------------------------------------------ |
| 1–7   | Data collection, annotation, frame extraction, dataset construction      |
| 8–12  | Stage 1a/1b/1c — CNN baselines (custom, ResNet-18, VGG-16)               |
| 13–20 | Stage 2 — CNN+LSTM                                                       |
| 21–25 | Analysis, ablations, error analysis (including pose modality comparison) |
| 26–30 | Writeup                                                                  |

---

## 13. Known Limitations & Risks

| Risk                                                   | Mitigation                                                                            |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Clip bleeding (punch A retraction into punch B windup) | Asymmetric window (10 before, 5 after); flag clips with close neighboring annotations |
| Class imbalance (straights likely overrepresented)     | Monitor annotation counts; apply class weighting if needed                            |
| Annotation subjectivity (peak extension frame)         | Clear protocol + calibration set if multiple annotators                               |
| Small dataset size                                     | Augmentation; frozen backbone; keep model complexity modest                           |
| Video-level leakage                                    | Enforced at split time; verified before any training run                              |
| MediaPipe keypoint failures (occlusion, motion blur)   | Inspect confidence scores per frame; interpolate or drop low-confidence clips         |
| 1080p bringing no meaningful accuracy gain             | Expected — treat as robustness test, not performance booster                          |
