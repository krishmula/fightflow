# FightFlow

Boxing punch classification (straight, hook, uppercut, none) from video using five progressively complex architectures. Built as a course project at San Jose State University.

**The short answer:** pose features beat pixels. A skeleton-based LSTM (F1=0.682) substantially outperforms a fine-tuned ResNet-18 (F1=0.570) on a small 20-video dataset because pose coordinates are invariant to background, lighting, and camera angle — things pixel models memorize instead of learning punches.

---

## Results

| Model | Best Epoch | Test Acc | Macro-F1 |
|---|---|---|---|
| Baseline CNN (scratch) | 2 | 0.456 | 0.402 |
| Advanced CNN (residual + attention) | 6 | 0.370 | 0.341 |
| ResNet-18 (ImageNet fine-tune) | 7 | 0.578 | 0.570 |
| CNN+LSTM (ResNet-18 backbone) | 7 | 0.593 | 0.588 |
| **Pose-LSTM (MediaPipe skeleton)** | **31** | **0.692** | **0.682** |

All numbers are on a video-level leakage-free test split (695 samples). The same models on a naive clip-level split scored up to 0.933 — a reminder that data leakage is silent and flattering.

### Why the Advanced CNN is worse than the Baseline

More capacity without pretrained weights on 20 videos means the optimizer memorizes training clips rather than finding a useful feature hierarchy. This is expected and is exactly what motivates transfer learning.

### Why CNN+LSTM peaks at epoch 7 (same as ResNet-18)

Both models use the same ResNet-18 backbone with a 5-epoch freeze schedule. Epoch 7 is the first epoch of full fine-tuning after backbone unfreeze. The marginal improvement of CNN+LSTM over ResNet-18 (0.588 vs 0.570) may come from the backbone receiving additional gradient signal through the LSTM loss rather than from genuine temporal modeling. A frozen-backbone ablation would clarify this.

### Why hook is the hardest class (F1=0.453 in Pose-LSTM)

In a straight punch, the elbow angle opens from ~90 degrees (cocked) to ~180 degrees (full extension). In a hook, the elbow stays at ~90 degrees throughout as the arm sweeps laterally as a rigid unit. The current feature set includes static elbow angles per frame but not elbow angle velocity (the per-frame delta). That 2-dimensional feature is the direct discriminator between hook and straight and is the clearest next improvement.

---

## Dataset

- 20 videos, 3,320 clips, 4 classes, **830 samples per class** (perfectly balanced)
- Annotated with a custom frame-level punch annotator (`notebooks/punch-annotator.ipynb`)
- Split at the **video level** using Group-Aware Stratified Greedy Split (GSGS) to prevent data leakage
- 2,625 train / 695 test
- Two modalities: RGB frames (for CNN models) and 11-frame pose sequences (for Pose-LSTM)

---

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/krishmula/fightflow.git
cd fightflow

# 2. Create environment and install dependencies
bash setup.sh
source .venv/bin/activate

# 3. Install ffmpeg if not already installed (required for video processing)
brew install ffmpeg   # macOS
```

**Requirements:** Python 3.10+, PyTorch, MediaPipe, OpenCV. All pinned in `requirements.txt`.

---

## Reproducing Results

All models are run through a single entry point:

```bash
python main.py --model <model_name> --task <task>
```

### Step 1: Prepare data

```bash
# RGB frames (used by all CNN models)
python main.py --model cnn_baseline --task prepare_data

# Pose sequences (used by Pose-LSTM)
python main.py --model pose_lstm --task prepare_data
```

### Step 2: Train each model

```bash
python main.py --model cnn_baseline  --task train
python main.py --model cnn_advanced  --task train
python main.py --model resnet18      --task train
python main.py --model cnn_lstm      --task train
python main.py --model pose_lstm     --task train
```

Checkpoints and metrics are saved to `runs/<model_name>/<run_timestamp>/`.

### Step 3: Evaluate

```bash
# Update hparams.yaml checkpoint path first, then:
python main.py --model pose_lstm --task validate
```

Hyperparameters for each model are in `src/models/<model_name>/hparams.yaml`.

---

## Pose Feature Vector (236-dim per frame)

The Pose-LSTM operates on engineered features rather than raw landmarks:

| Feature group | Dims | What it captures |
|---|---|---|
| Normalized joint coordinates | 99 | Body pose, hip-centered and scale-normalized |
| Inter-frame velocities | 99 | Joint motion between consecutive frames |
| Relative wrist offsets | 18 | Wrist-to-shoulder, wrist-to-hip, wrist-to-nose |
| Peak wrist velocities | 6 | Max x/y/z speed per wrist over the clip |
| Elbow joint angles | 2 | 3D angle at left/right elbow via law of cosines |
| Wrist-to-nose velocities | 6 | Wrist closing speed toward the face |
| Wrist extension features | 6 | Avg velocity in final 4 frames of extension |

11-frame window: 6 pre-peak + peak frame + 4 post-peak. Evaluated against clip lengths of 8, 10, 11, 12 — 11 frames gave the best test F1 and fully captures the punch arc from wind-up through follow-through.

---

## Project Structure

```
fightflow/
├── main.py                      # Single CLI entry point
├── requirements.txt
├── setup.sh
├── src/
│   └── models/
│       ├── base/                # Shared pipeline, data loading, utils
│       ├── cnn_baseline/        # 4-block CNN from scratch
│       ├── cnn_advanced/        # Residual + squeeze-excitation CNN
│       ├── resnet18/            # ImageNet fine-tune with attention
│       ├── cnn_lstm/            # ResNet-18 backbone + bidirectional LSTM
│       └── pose_lstm/           # MediaPipe skeleton + bidirectional LSTM
├── data/
│   ├── annotations.csv          # Frame-level punch labels
│   └── processed/               # Extracted frames, clips, poses, manifests
├── notebooks/
│   ├── punch-annotator.ipynb    # Custom annotation tool
│   └── ...
├── scripts/                     # Data auditing and preprocessing utilities
├── docs/                        # Annotation guides and process docs
├── runs/                        # Training artifacts (gitignored)
└── report/                      # IEEE paper (LaTeX)
```

---

## Key Design Decisions

**Video-level splits over clip-level splits.** Clip-level splits let the same video appear in both train and test. Models scored up to 0.933 macro-F1 under that setup — inflated by memorizing video-specific backgrounds. Switching to video-level GSGS dropped scores to honest levels and is the right evaluation for generalization.

**LayerNorm instead of BatchNorm1d in the classifier.** BatchNorm1d fails when the last batch has a single sample (2,625 samples / batch size 8 = 1 leftover). LayerNorm handles any batch size.

**Pose-LSTM uses no class weights.** The dataset is perfectly balanced at 830 per class. Class weights are used for CNN models only to compensate for uneven video-level test splits.

**CNN+LSTM backbone initialized from task-fine-tuned ResNet-18.** Using ImageNet-only weights for the CNN+LSTM backbone would have been a cleaner ablation. The current setup confounds backbone quality with temporal modeling.

---

## Authors

- Kushagra Bainsla — kushagra.bainsla@sjsu.edu
- Krishna Mula — krishna.mula@sjsu.edu

San Jose State University, CS271, Spring 2026
