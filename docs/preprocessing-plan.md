# Data Preprocessing Plan for Boxing Technique Analysis

This README is a **full, beginner-friendly plan** for the **data preprocessing stage** of the project.

It is written to answer one specific question:

> How do we go from raw dataset files to a clean, trustworthy, model-ready dataset of labeled clips?

This plan expands the project proposal and broader roadmap into a concrete operating guide for this exact stage.

---

## 1. What this stage is trying to accomplish

By the end of preprocessing, we want to have a dataset that looks like this:

```text
One training example =
  short clip of video frames
  + label (jab / cross / hook / uppercut, etc.)
  + metadata (video id, start frame, end frame, split, quality flags)
  + optional aligned skeleton/keypoint data
```

In other words, the raw dataset might start as:

```text
YouTube links
+ annotation spreadsheets
+ skeleton/keypoint .npy files
```

And we want to end with:

```text
clean_manifest.csv
train.csv
val.csv
test.csv
processed RGB clips
processed skeleton clips
quality reports and sanity-check visualizations
```

That is the entire purpose of this stage.

---

## 2. What we currently have

From your dataset structure, we currently have:

```text
.
├── Annotation_files
│   ├── V1.xlsx
│   ├── V2.xlsx
│   ├── ...
│   └── V10.xlsx
├── RGB_videos
│   └── Meta_data.ods
└── Skeleton_data
    ├── V1.npy
    ├── V2.npy
    ├── ...
    └── V10.npy
```

### What each part probably means

#### `Annotation_files/V1.xlsx ... V10.xlsx`
These are spreadsheets containing the labels.

Each row likely represents one annotated punch event, such as:

- when the punch starts
- when the punch ends
- what class it is

Example idea:

| start_frame | end_frame | class |
|---|---:|---|
| 1240 | 1258 | Jab |
| 3372 | 3391 | Cross |

#### `RGB_videos/Meta_data.ods`
This likely contains information about the original videos, probably including:

- YouTube URLs
- maybe titles
- maybe FPS / IDs / notes

This is important because the raw video files themselves may not yet be downloaded.

#### `Skeleton_data/V1.npy ... V10.npy`
These are NumPy arrays containing pose/keypoint data.

Very likely, each array stores body joint coordinates for each annotated clip or each frame.

Typical possibilities:

```python
(num_events, num_frames, num_joints, 2)
```

or

```python
(num_frames, num_people, num_joints, 3)
```

where the last dimension might be:

- `(x, y)`
- or `(x, y, confidence)`

We must inspect this instead of guessing.

---

## 3. The most important idea in this whole stage

The most important thing is **alignment**.

We are not just collecting files.
We are verifying that:

```text
annotation timing
        matches
video timing
        matches
skeleton timing
```

### Why alignment matters

If the annotation says “jab” but the extracted frames show the tail end of a hook, then the label is wrong for training.

That creates **label noise**, and the model learns the wrong thing.

So the core job is:

```text
For each labeled event:
1. know exactly where it is in the video
2. know exactly which frames belong to it
3. know exactly whether the skeleton data matches those same frames
4. only keep it if it is trustworthy
```

---

## 4. What you need to understand before starting

This section explains the basic concepts from scratch.

### 4.1 What is a frame?
A video is a sequence of still images.
Each still image is called a **frame**.

If a video is 30 FPS, that means:

- 30 frames are shown every second

So:

- frame 0 = beginning
- frame 30 ≈ 1 second
- frame 60 ≈ 2 seconds

### 4.2 What is FPS?
**FPS** = frames per second.

This matters because annotations may be in:

- frame numbers
- seconds
- timestamps

If the annotation is in seconds, then we convert to frames using:

```text
frame_index = seconds × fps
```

If the annotation is already in frames, we must still verify that the downloaded video’s frame count and FPS are compatible.

### 4.3 What is a clip?
A **clip** is a short sequence of consecutive frames.

For example:

```text
Frames 1000 to 1015 = 16-frame clip
```

This clip might contain a single punch event.

### 4.4 What is a label?
A **label** is the class name for the action in the clip.
Examples:

- Jab
- Cross
- Lead Hook
- Rear Hook
- Lead Uppercut
- Rear Uppercut

Sometimes we may merge left/right or lead/rear variants into broader categories.

### 4.5 What is pose / skeleton data?
Pose estimation turns a person into body keypoints.

For example:

```text
head
shoulders
elbows
wrists
hips
knees
ankles
```

Instead of storing the image itself, the file stores approximate coordinates like:

```python
[
  [x1, y1],
  [x2, y2],
  ...
]
```

for each frame.

That means a skeleton file can capture motion very efficiently.

---

## 5. What success looks like

Preprocessing is successful only if all of the following are true:

- we can parse every annotation file
- we know what every column means
- we know what every skeleton file shape means
- we can download or locate every source video
- we can map each annotation to an exact video segment
- we can visually verify random samples
- we can create a clean manifest of all usable events
- we can split data into train/val/test without leakage
- we can load processed examples directly into a PyTorch dataset later

If any of those are missing, preprocessing is not actually complete.

---

## 6. High-level roadmap for this stage

Here is the full roadmap before we zoom into each part.

```text
Step 0  Prepare workspace
Step 1  Audit the raw files
Step 2  Understand the annotation format exactly
Step 3  Understand the skeleton format exactly
Step 4  Download the source videos
Step 5  Verify video metadata and frame counts
Step 6  Build a unified manifest
Step 7  Check alignment between annotations, video, and skeletons
Step 8  Decide label mapping
Step 9  Clean bad or ambiguous samples
Step 10 Extract fixed-length RGB clips
Step 11 Process skeleton clips
Step 12 Build train/val/test splits
Step 13 Save processed artifacts
Step 14 Run final sanity checks
Step 15 Freeze the preprocessing output
```

That is the real end-to-end preprocessing flow.

---

## 7. Step 0 — Prepare the workspace

Before touching the dataset, create a clean project structure.

### Recommended folder layout

```text
project_root/
├── data/
│   ├── raw/
│   │   ├── Annotation_files/
│   │   ├── RGB_videos/
│   │   └── Skeleton_data/
│   ├── downloaded_videos/
│   ├── interim/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
├── reports/
└── README_preprocessing_plan.md
```

### Why this matters

We want to separate:

- **raw** data we never edit
- **interim** files created during inspection or conversion
- **processed** files used for modeling

This prevents accidental corruption of the original dataset.

### Environment setup

At minimum, install tools for:

- Python
- pandas
- numpy
- openpyxl
- opencv-python
- matplotlib
- ffmpeg
- yt-dlp
- jupyter

Example:

```bash
pip install pandas numpy openpyxl opencv-python matplotlib jupyter yt-dlp
```

Also make sure `ffmpeg` is installed on the system.

---

## 8. Step 1 — Audit the raw files

This is the very first real task.

Do **not** start by processing everything.
Start by inspecting what the files actually contain.

### 8.1 Count the files
Verify:

- 10 annotation files exist
- 10 skeleton files exist
- metadata file exists

### 8.2 Open one annotation file manually
Start with `V1.xlsx`.

Check:

- what are the column names?
- are there header rows?
- are the times stored as frames or seconds?
- are labels clean strings?
- are there empty rows?
- are there duplicate rows?

### 8.3 Load one skeleton file manually
Start with `V1.npy`.

Check:

- array shape
- data type
- min/max values
- whether values look normalized
- whether there are all-zero frames

### 8.4 Open metadata file manually
Look inside `Meta_data.ods` and identify:

- video ids
- URLs
- any title or duration columns
- any FPS or resolution columns

### Output of Step 1
By the end of the audit, you should have a small written note answering:

- what each file type contains
- how annotations are represented
- how skeletons are represented
- where the source videos come from

If you cannot answer those yet, do not move on.

---

## 9. Step 2 — Understand the annotation format exactly

This step deserves its own section because annotations control everything downstream.

### 9.1 Determine the exact schema
For each row, identify:

- event identifier, if any
- start position
- end position
- label/class
- optional boxer identity, if present
- optional notes or quality flags

### 9.2 Determine the time unit
This is crucial.

You must know whether `start` and `end` are:

- frame indices
- seconds
- timestamps like `00:01:23.400`

You cannot safely extract clips until this is confirmed.

### 9.3 Determine the label vocabulary
Make a list of all unique labels across all 10 files.

Example:

```text
Jab
Cross
Lead Hook
Rear Hook
Lead Uppercut
Rear Uppercut
```

Also check for label inconsistencies such as:

- `jab` vs `Jab`
- `LeadHook` vs `Lead Hook`
- trailing spaces
- typos

### 9.4 Measure event duration
For every row, compute:

```text
duration = end - start + 1
```

Then summarize:

- minimum duration
- maximum duration
- mean duration
- median duration
- per-class duration statistics

### Why this matters

This tells us:

- how long a typical punch event lasts
- whether clip extraction should use padding or resampling
- whether there are annotation errors

### 9.5 Detect broken rows
Flag rows where:

- start is missing
- end is missing
- class is missing
- end < start
- duration is absurdly small or absurdly large

### Output of Step 2
Create an audit table or CSV with columns like:

```text
video_id
num_rows
unique_labels
min_duration
max_duration
bad_rows
```

---

## 10. Step 3 — Understand the skeleton format exactly

This is the second most important inspection task.

### 10.1 Load each `.npy` file and record its shape
For every file `V1.npy` through `V10.npy`, record:

- array shape
- dtype
- min/max value
- percentage of zeros

### 10.2 Infer what each dimension means
For example, if a file has shape:

```python
(N, 25, 17, 2)
```

then the likely interpretation is:

- `N` = number of events
- `25` = frames per event
- `17` = body joints
- `2` = x, y coordinates

But do not assume this permanently until confirmed against annotations.

### 10.3 Check whether skeleton count matches annotation count
For each video:

```text
number of annotation rows ?= first dimension or logical event count in the skeleton file
```

If they match exactly, that is a strong sign that skeleton clips are already aligned per annotation.

If they do not match, we need deeper investigation.

### 10.4 Check whether skeleton coordinates are normalized
Inspect value ranges.

Common possibilities:

- `0 to 1` normalized coordinates
- pixel coordinates like `0 to 1920`
- coordinates plus confidence score

### 10.5 Check for missing detections
Common failure pattern:

```python
[[0, 0], [0, 0], ..., [0, 0]]
```

for a frame where pose estimation failed.

Measure:

- percentage of all-zero frames per event
- percentage of events with too many missing frames

### 10.6 Visualize a few skeleton clips
Plot some keypoints frame by frame.

You want to answer:

- does the skeleton look human?
- does motion look plausible?
- are there obvious detection failures?
- is the punch arm track visible?

### Output of Step 3
Create a skeleton audit report containing:

- shapes per file
- inferred meaning of each dimension
- missing-data rates
- match / mismatch with annotation counts
- a few saved visualization images or GIFs

---

## 11. Step 4 — Download the source videos

Now that we understand the metadata, download the videos.

### 11.1 Why this is necessary
Even if the skeleton data is available, we still need the raw videos because:

- the CNN baseline uses RGB frames
- the CNN+LSTM model uses RGB clips
- we need visual verification of annotations
- we may need videos later for demos and VLM integration

### 11.2 Extract YouTube URLs from metadata
From `Meta_data.ods`, build a table like:

| video_id | url |
|---|---|
| V1 | ... |
| V2 | ... |

### 11.3 Download carefully
Use `yt-dlp` and save files with stable names.

Example naming:

```text
data/downloaded_videos/V1.mp4
...
data/downloaded_videos/V10.mp4
```

### 11.4 Record download metadata
For each file, record:

- final filename
- resolution
- FPS
- duration
- frame count if possible
- any warnings or missing videos

### Risks
YouTube videos can be:

- removed
- region blocked
- re-encoded
- downloaded at a different FPS than expected

So after download, verification is mandatory.

### Output of Step 4
A CSV like:

```text
video_id,file_path,fps,duration_sec,num_frames,download_status
```

---

## 12. Step 5 — Verify video metadata and frame counts

This is where we connect annotations to the real video files.

### 12.1 Read video properties programmatically
For each video, extract:

- FPS
- total frame count
- width
- height
- duration

### 12.2 Compare annotation ranges to video length
If annotations are frame-based, verify:

```text
max annotated end_frame <= total frame count - 1
```

If this is false, something is wrong.

### 12.3 Convert seconds to frames if needed
If annotations are in seconds, convert them using the downloaded video’s FPS.

### 12.4 Spot-check a few annotated events visually
Pick a few rows from `V1.xlsx`.
Extract the corresponding frames from the video.
Check whether the labeled punch appears in that interval.

This is the first true alignment test.

### Output of Step 5
A validation note per video saying:

- annotation ranges are valid or not
- time unit confirmed
- FPS confirmed
- any video that does not align is flagged

---

## 13. Step 6 — Build a unified manifest

This is the central data table for the whole project.

### What is a manifest?
A **manifest** is one clean table where each row represents one candidate training example.

### Recommended manifest columns

| column | meaning |
|---|---|
| sample_id | unique ID for the event |
| video_id | source video ID |
| source_video_path | path to downloaded video |
| ann_row_index | original row number in spreadsheet |
| start_frame | event start |
| end_frame | event end |
| duration_frames | event length |
| class_raw | original label string |
| class_clean | cleaned label |
| class_id | numeric class id |
| split | train / val / test later |
| skeleton_path | source skeleton file |
| skeleton_index | matching event index if aligned |
| has_video | yes/no |
| has_skeleton | yes/no |
| quality_flag | ok / review / reject |
| notes | optional comments |

### 13.1 Clean labels while building the manifest
Standardize label strings.

For example:

- trim whitespace
- unify capitalization
- fix obvious variants

### 13.2 Assign stable sample IDs
Example:

```text
V1_00001
V1_00002
...
V10_00543
```

### 13.3 Preserve raw information
Never throw away the raw label or original row index.
Those fields help debugging later.

### Output of Step 6
Create:

```text
data/processed/dataset_manifest.csv
```

This file becomes the source of truth for preprocessing.

---

## 14. Step 7 — Verify alignment between annotations, video, and skeletons

This is the most important validation stage.

We must confirm that all three modalities refer to the same event.

```text
annotation row k
     ↕
video event k
     ↕
skeleton event k
```

### 14.1 First test: row count alignment
If `V1.xlsx` has 500 rows and `V1.npy` has 500 event clips, that is a good sign.

### 14.2 Second test: visual alignment
For a selected event:

1. extract RGB frames from the annotated video interval
2. render the matching skeleton clip
3. compare them side by side

You want to see whether the pose motion matches the person motion in the video.

### 14.3 Third test: temporal alignment
Check if the punch motion occurs during the same relative frames in both views.

Bad case:

```text
video shows punch starting at frame 8
skeleton peak motion occurs at frame 2
```

That suggests an offset problem.

### 14.4 Sample across classes and videos
Do not test only one clip.
Check:

- at least 3 to 5 random events per class
- at least several different videos

### 14.5 Decide alignment status per video
For each video, classify alignment as:

- verified good
- uncertain, needs manual review
- bad, not usable until fixed

### Output of Step 7
A short alignment report with screenshots or GIFs.
Without this report, preprocessing is incomplete.

---

## 15. Step 8 — Decide the label mapping

Now that the raw labels are understood, decide what the model will actually predict.

### Option A — 4 merged classes
Map:

- Jab → Jab
- Cross → Cross
- Lead Hook + Rear Hook → Hook
- Lead Uppercut + Rear Uppercut → Uppercut

### Option B — 6 fine-grained classes
Keep:

- Jab
- Cross
- Lead Hook
- Rear Hook
- Lead Uppercut
- Rear Uppercut

### Which one should we choose first?
Start with the simpler option unless the class counts are strong and balanced enough for 6 classes.

### Why merging may help
Because the first version of the model should solve a learnable problem.
If left/right or lead/rear variants are too subtle for the available data, merging reduces confusion and gives a cleaner first milestone.

### Output of Step 8
Create:

```text
data/processed/label_map.json
```

Example:

```json
{
  "Jab": 0,
  "Cross": 1,
  "Hook": 2,
  "Uppercut": 3
}
```

Also store a mapping from raw labels to clean labels.

---

## 16. Step 9 — Clean bad or ambiguous samples

Not every annotated event should be kept.
This step improves label quality.

### 16.1 Filter obvious annotation problems
Reject rows where:

- start or end is missing
- end < start
- duration too short to contain the action
- duration too long and likely includes multiple actions

### 16.2 Filter poor skeleton samples
Reject or flag events where:

- too many frames are all-zero
- keypoints jump unrealistically
- main fighter is missing most of the clip

### 16.3 Filter poor RGB samples
Reject or flag events where:

- fighter is fully occluded
- punch is not visible
- camera cut happens mid-event
- replay / slow-motion clip is mixed in unexpectedly

### 16.4 Handle overlapping annotations
Two labels close together may create ambiguous clips.

Decide a policy:

- keep only if overlap is small
- drop if multiple punches overlap heavily
- mark for manual review

### 16.5 Use quality flags instead of immediate deletion when unsure
Recommended quality flags:

- `ok`
- `review`
- `reject`

This is better than throwing data away too early.

### Output of Step 9
Updated manifest with quality decisions.
Only `ok` rows will feed the first training dataset.

---

## 17. Step 10 — Extract fixed-length RGB clips

Now we create model-ready RGB examples.

### Why fixed-length clips are needed
Neural models are much easier to train when every sample has the same temporal length.

Example target lengths:

- 8 frames
- 16 frames
- 25 frames
- 32 frames

### 17.1 Choose an initial clip length
A very reasonable starting point is 16 or 25 frames, depending on what best matches the annotation and skeleton structure.

### 17.2 Decide how to sample from variable event durations
If an event lasts exactly the target length, easy.

If it is longer, sample frames evenly.
If it is shorter, pad or expand a window around it.

Common strategy:

```text
1. find event center
2. create a window of fixed size around the center
3. clamp to video boundaries
```

Alternative strategy:

```text
sample uniformly from start_frame to end_frame
```

### 17.3 Extract frames
For each manifest row marked `ok`:

- open the source video
- grab the selected frame indices
- save them as images or pack into arrays

### 17.4 Apply spatial preprocessing
Per frame:

1. resize
2. crop
3. convert to tensor later
4. normalize using ImageNet mean/std

Typical shape for one clip:

```python
(T, 3, 224, 224)
```

### 17.5 Save output
Two reasonable storage choices:

#### Option 1: save individual image frames

```text
data/processed/rgb_clips/V1_00001/frame_000.jpg
...
```

#### Option 2: save tensor arrays

```text
data/processed/rgb_arrays/V1_00001.npy
```

For debugging, frame folders are easier.
For speed, arrays are easier.

### Output of Step 10
A reproducible RGB clip extraction pipeline and processed RGB clip data.

---

## 18. Step 11 — Process skeleton clips

Now create model-ready skeleton inputs.

### 18.1 Decide whether skeletons are already clip-aligned
If the data shape is already `(N_events, T, J, C)`, then most of the work may already be done.

We still need to standardize and clean it.

### 18.2 Handle missing keypoints
Possible policies:

- keep zeros and provide a mask
- interpolate missing frames
- forward-fill from nearby valid frames
- reject very broken clips

For the first version, a simple and transparent policy is best.

### 18.3 Normalize coordinates
Possible normalizations:

- divide by image width/height if in pixels
- center by torso or hip midpoint
- scale by torso size or person height

Why?
Because the model should learn motion pattern, not camera location.

### 18.4 Flatten or preserve structure
Two common formats:

#### Flattened for simple LSTM

```python
(T, 17, 2) -> (T, 34)
```

#### Structured for more advanced models

```python
(T, 17, 2)
```

For a first LSTM baseline on skeletons, flattened format is fine.

### 18.5 Save processed skeleton clips
Example:

```text
data/processed/skeleton_clips/V1_00001.npy
```

### Output of Step 11
A clean skeleton dataset that can be loaded clip-by-clip.

---

## 19. Step 12 — Build train / validation / test splits

This is where many projects accidentally leak information.

### The rule
Do **not** split randomly at the frame level.
Do **not** split randomly at the clip level if clips from the same source video can appear in multiple splits.

That would let the model see almost the same context during training and testing.

### Recommended policy
Split by **video ID**.

Example:

- Train: `V1` to `V6`
- Validation: `V7`, `V8`
- Test: `V9`, `V10`

### Why this is better
It tests generalization to unseen fight footage instead of memorizing the same video characteristics.

### 19.1 Check class balance per split
For each split, count:

- number of samples
- number of samples per class
- percent of total per class

### 19.2 Handle imbalance if needed
Do not oversolve this during preprocessing, but record it.

Possible later strategies:

- weighted loss
- class-balanced sampling
- augmentation of minority classes

### Output of Step 12
Create:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Each should be a filtered subset of the manifest.

---

## 20. Step 13 — Save all processed artifacts cleanly

At this point, save everything in a stable, reproducible format.

### Recommended processed directory

```text
data/processed/
├── dataset_manifest.csv
├── label_map.json
├── train.csv
├── val.csv
├── test.csv
├── rgb_clips/
│   ├── V1_00001/
│   └── ...
├── skeleton_clips/
│   ├── V1_00001.npy
│   └── ...
├── reports/
│   ├── annotation_audit.csv
│   ├── skeleton_audit.csv
│   ├── alignment_report.md
│   └── class_distribution.csv
└── visual_checks/
    ├── sample_rgb/
    ├── sample_skeleton/
    └── overlay_checks/
```

### Why this matters
This lets you restart training later without rebuilding the entire dataset from scratch.

---

## 21. Step 14 — Run final sanity checks

Before declaring preprocessing done, run a final checklist.

### 21.1 Manifest integrity checks
Confirm:

- every `sample_id` is unique
- every `start_frame <= end_frame`
- every kept sample has a valid class id
- every kept sample points to an existing file

### 21.2 Visual sanity checks
Randomly inspect samples from each class and each split.

Ask:

- does the clip actually show the labeled action?
- are hooks really hooks?
- are uppercuts visible?
- are padding and crops reasonable?

### 21.3 Split integrity checks
Confirm:

- no video id appears in more than one split
- no sample appears twice
- class counts are reasonable

### 21.4 Modality alignment checks
For random samples that have both RGB and skeletons:

- do they correspond to the same event?
- do they have the same label?
- does motion timing look consistent?

### Output of Step 14
A short `preprocessing_validation.md` report.

---

## 22. Step 15 — Freeze the preprocessing output

Once the sanity checks pass:

- do not keep modifying processed files manually
- version the manifest and label mapping
- record the exact script versions used

This matters because model results are only meaningful if the dataset version is stable.

---

## 23. The exact order I recommend you do things in practice

This section is the operational order, not just the conceptual order.

### Phase A — Understand one video completely
Start with **only V1**.

1. inspect `V1.xlsx`
2. inspect `V1.npy`
3. inspect V1 metadata row
4. download V1 video
5. verify annotation timing against V1 video
6. compare one or more V1 skeleton samples to the video
7. create a tiny manifest for V1
8. extract a few RGB clips
9. plot a few skeleton clips

Why start with one video?
Because if the logic is wrong, it is much cheaper to discover that on one file than on all ten.

### Phase B — Generalize to all videos
Once V1 works:

1. run the same audit for V2 to V10
2. build the full manifest
3. generate reports
4. decide labels
5. process clips in bulk
6. split the data

### Phase C — Lock the first usable dataset
Only after the above is validated:

1. finalize `dataset_manifest.csv`
2. finalize label mapping
3. finalize train/val/test splits
4. freeze processed outputs

---

## 24. Recommended scripts to implement during this stage

These scripts are enough to cover the full preprocessing pipeline.

### `scripts/audit_annotations.py`
Purpose:

- read all `.xlsx` files
- summarize labels and durations
- report broken rows

### `scripts/audit_skeletons.py`
Purpose:

- read all `.npy` files
- record shapes and missing-data rates
- visualize random events

### `scripts/parse_metadata.py`
Purpose:

- read `Meta_data.ods`
- extract video URLs and ids

### `scripts/download_videos.py`
Purpose:

- download videos with stable filenames
- record video metadata

### `scripts/build_manifest.py`
Purpose:

- merge annotations, video info, and skeleton references into one CSV

### `scripts/check_alignment.py`
Purpose:

- sample events
- render RGB clips and skeleton clips side by side
- help confirm alignment

### `scripts/extract_rgb_clips.py`
Purpose:

- create fixed-length RGB clips from the manifest

### `scripts/process_skeletons.py`
Purpose:

- normalize, clean, and save skeleton clips

### `scripts/make_splits.py`
Purpose:

- assign train/val/test by video ID
- save split CSVs

### `scripts/validate_processed_dataset.py`
Purpose:

- run final integrity checks

---

## 25. Common mistakes to avoid

### Mistake 1 — Download all videos first and inspect later
Better approach:

- inspect one video deeply first

### Mistake 2 — Assume annotation units without checking
Better approach:

- confirm whether they are frames or seconds

### Mistake 3 — Trust skeleton files blindly
Better approach:

- verify shape, alignment, and missing data

### Mistake 4 — Random clip-level split
Better approach:

- split by source video or fight

### Mistake 5 — Over-clean too early
Better approach:

- use quality flags so uncertain samples can be reviewed later

### Mistake 6 — Lose traceability
Better approach:

- keep raw labels, original row indices, and file links in the manifest

---

## 26. What the final deliverables of this stage should be

At the end of preprocessing, you should have all of the following.

### Required files

- `dataset_manifest.csv`
- `label_map.json`
- `train.csv`
- `val.csv`
- `test.csv`
- processed RGB clips or arrays
- processed skeleton clips or arrays

### Required reports

- annotation audit report
- skeleton audit report
- alignment report
- class distribution report
- preprocessing validation report

### Required visual checks

- random RGB clips per class
- random skeleton plots per class
- at least a few RGB-vs-skeleton comparisons

If those are not produced, this stage is still incomplete.

---

## 27. Clear definition of done

You are done with preprocessing only when you can say:

1. **I know what every raw file means.**
2. **I verified that annotations map correctly to the videos.**
3. **I verified whether skeleton data is aligned and usable.**
4. **I created a single clean manifest containing all usable events.**
5. **I chose a label mapping and encoded labels numerically.**
6. **I removed or flagged low-quality samples.**
7. **I created fixed-length RGB and/or skeleton clips.**
8. **I built train/val/test splits without leakage.**
9. **I ran sanity checks and visual inspections.**
10. **I saved the processed dataset in a stable format for training.**

That is what “done” means for this step.

---

## 28. The practical first-week plan for this preprocessing stage

If you want this reduced into the exact first actions to take, do them in this order:

### Day 1
- inspect `V1.xlsx`
- inspect `V1.npy`
- inspect `Meta_data.ods`
- write down the discovered schemas

### Day 2
- download V1 video
- verify V1 annotation timing visually
- compare V1 RGB and skeleton alignment

### Day 3
- write annotation audit script for all videos
- write skeleton audit script for all videos
- collect class counts and shape summaries

### Day 4
- build initial full `dataset_manifest.csv`
- clean labels
- create `label_map.json`

### Day 5
- decide 4-class or 6-class setup
- add quality flags
- reject clearly broken samples

### Day 6
- implement RGB clip extraction
- implement skeleton processing

### Day 7
- build train/val/test split files
- run final validation checks
- freeze first usable processed dataset

---

## 29. Bottom line

The first thing to do is **not** “download everything and hope it lines up.”

The correct mindset is:

```text
inspect
understand
verify alignment
build manifest
clean
extract
split
validate
freeze
```

That is the full preprocessing stage.

Once this is done well, the modeling stage becomes much easier and much more trustworthy.

---

## 30. After this stage

Once preprocessing is complete, the next stage is:

```text
Dataset loader
→ CNN baseline
→ CNN+LSTM model
→ evaluation
```

But do not rush there.
A good model cannot fix a bad preprocessing pipeline.

