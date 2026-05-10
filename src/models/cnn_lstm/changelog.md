# CNN-LSTM Training Changelog

## [2026-05-08] - Initial Baseline with Optimized ResNet Backbone

### Summary of Run: `cnn_lstm_20260508_202107`
*   **Backbone**: ResNet-18 (Fine-tuned from `resnet18_multiclass_20260508_191234`)
*   **Technique**: 5-epoch frozen backbone warmup, followed by full fine-tuning.
*   **Results**: 
    *   Peak Val Macro-F1: **0.698** (Epoch 14).
    *   Validation Accuracy: ~71%.
    *   Training Macro-F1: 0.98 (Significant overfitting).
*   **Observation**: 
    *   Successfully outperformed the pure ResNet baseline (0.66 -> 0.698).
    *   Experienced catastrophic instability at Epoch 17 (Val loss spiked to 3.5).
    *   High variance in validation metrics suggests the model is very sensitive to learning rate and specific sequence patterns.

## [2026-05-09] - Stability Improved, but Regularization too Aggressive

### Summary of Run: `cnn_lstm_20260509_214918`
*   **Technique**: Gradient Clipping (1.0), Split LR (1e-5 backbone, 1e-4 LSTM), Higher Regularization (WD 0.001, Dropout 0.5).
*   **Results**:
    *   Peak Val Macro-F1: **0.57** (Epoch 1).
    *   Validation metrics degraded after unfreezing at Epoch 6.
*   **Observation**:
    *   **Stability**: Loss spikes are gone! Gradient clipping successfully stabilized the loss landscape.
    *   **Regression**: Overall performance dropped significantly. The high Weight Decay (0.001) and early unfreezing (Epoch 6) seem to be preventing the model from converging on the optimal motion features.

## [2026-05-10] - The "Locked Backbone" Pivot

### Summary of Run: `cnn_lstm_20260509_232125` (Goldilocks)
*   **Technique**: freeze_epochs 10, batch_size 16, WD 0.0005.
*   **Results**:
    *   Peak Val Macro-F1: **0.59** (Epoch 9 - during warmup).
    *   Performance crashed immediately upon unfreezing at Epoch 11.
*   **Observation**:
    *   **Backbone Sensitivity**: Unfreezing the backbone on a dataset this small (8 train videos) is counter-productive. The model immediately begins memorizing backgrounds rather than refining motion.
    *   **Performance Bottleneck**: The best performance was consistently achieved while the backbone was frozen.

## [2026-05-10] - The "Partial Unfreeze" Surgical Strike

### Summary of Run: `cnn_lstm_20260510_005721` (Fixed Feature)
*   **Technique**: Permanent Backbone Freeze, clip_length 8, lstm_hidden 512.
*   **Results**:
    *   Peak Val Macro-F1: **0.597** (Epoch 4).
    *   Plateaued very early; frozen features lacked the flexibility to capture high-speed motion nuances.
*   **Observation**:
    *   **Feature Rigidity**: Treating the ResNet as a purely fixed extractor is too restrictive. The model needs to "tune its eyes" slightly to the specific motion patterns of this dataset.
    *   **Subsampling Loss**: At 8 frames, we might be losing too much temporal information for a movement that is roughly 10 frames long.

## [2026-05-10] - The "Temporal Consistency" Breakthrough

### Summary of Run: `cnn_lstm_20260510_112536` (Surgical Strike)
*   **Technique**: unfreeze_partial(), clip_length 10, lstm_hidden 512.
*   **Results**:
    *   Peak Val Macro-F1: **0.547** (Epoch 3).
    *   Severe regression after unfreezing; validation metrics crashed to the 0.30s.
*   **Critical Bug Found: "Augmentation Jitter"**:
    *   The `ClipDataset` was applying *different* random transforms to each frame in a sequence. This effectively "shredded" the temporal motion features, making it impossible for the LSTM to learn smooth trajectories.

### Summary of Run: `cnn_lstm_20260510_123010` (Consistency Breakthrough)
*   **Technique**: Synchronized Augmentation, freeze_epochs 10, clip_length 10.
*   **Results**:
    *   Peak Val Macro-F1: **0.6266** (Epoch 12).
    *   The model reached this peak shortly after unfreezing, confirming that consistent motion patterns allow the backbone to be fine-tuned effectively.
*   **Observation**:
    *   **Motion Recovery**: Fixing the "Jitter Bug" successfully restored the model's ability to learn temporal features.
    *   **Learning Momentum**: The model was still improving when it hit the evaluation crash, suggesting higher potential with more epochs.

### Planned Optimizations (Next Run - "Golden Sync" Strategy)
*   **Synchronized Augmentation**: Maintain the fix that ensures consistent motion across clips.
*   **Golden Context**: Revert to `clip_length: 12` to provide the LSTM with more temporal context.
*   **Fast Unfreeze**: Revert to `freeze_epochs: 5` to allow the backbone to adapt earlier.
*   **Extended Horizon**: Increase `epochs` to **50** and `patience` to **15** for full convergence.
*   **Stability**: Maintain `max_grad_norm: 5.0` for safe but aggressive learning.

---

## [2026-05-10] - "Golden Sync" Run (clip_length=12, freeze_epochs=5)

### Summary of Run: `cnn_lstm_20260510_143135`
*   **Backbone**: ResNet-18 (`cnn_init: latest`), Bidirectional LSTM (hidden=512, layers=1, output=mean).
*   **Config**: `clip_length: 12`, `freeze_epochs: 5`, `lr: 1e-4`, `patience: 15`, `batch_size: 16`.
*   **Dataset**: 1635 train clips / 725 val clips (80/20 split, no test set).
*   **Results**:
    *   Peak Val Macro-F1: **0.6154** at Epoch 3 (during frozen warmup phase).
    *   Final Val Accuracy: 62.2% | Val Loss: 1.204.
    *   Early stopping triggered at Epoch 18 (15 epochs without improvement).
*   **Per-Class Performance (Best Epoch)**:
    | Class | Precision | Recall | F1 |
    |---|---|---|---|
    | straight | 0.656 | 0.656 | 0.656 |
    | hook | 0.350 | 0.661 | **0.457** (weakest) |
    | uppercut | 0.736 | 0.649 | 0.690 |
    | none | 0.844 | 0.540 | 0.659 |
*   **Observation**:
    *   Performance peaked during the frozen warmup phase (Epoch 3) and collapsed upon backbone unfreezing at Epoch 6 - same pattern as prior runs.
    *   **`hook` class** is the primary bottleneck: low precision (0.35) due to 62 `none` and 61 `uppercut` frames being misclassified as `hook`.
    *   Val loss becomes highly volatile post-unfreeze (spikes to 2.29 - 3.20), indicating the backbone LR (2e-5) is still too aggressive for this dataset size.
    *   Training Macro-F1 continued climbing to **0.965** by Epoch 18, confirming severe overfitting after unfreezing.
    *   Did **not** reach the 0.70 Macro-F1 target.
