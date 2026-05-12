# ResNet-18 Model Changelog

All notable changes, experimental results, and optimization notes for the ResNet-18 punch classifier are documented here.

## [2026-05-12] - Breaking Overfitting: Regularization & Split LR

### Analysis of Run: `resnet18_multiclass_20260512_134205`
*   **Result**: High stability. **Best Macro-F1: 0.5806** at Epoch 28.
*   **Stability**: Unlike previous runs, this run showed no "unfreeze shock" or performance collapse at Epoch 6.
*   **Success**: **Split Learning Rate** (Backbone 1e-5, Head 5e-4) successfully protected the pretrained features during the transition to full fine-tuning.
*   **Success**: **RandomResizedCrop** and increased **Weight Decay (0.01)** successfully "broke" the model's ability to cheat by memorizing backgrounds, yielding an "honest" baseline for temporal modeling.
*   **Verdict**: While the final F1 (0.58) is numerically lower than the "cheating" run (0.65), this model is significantly more robust and is the selected candidate for the CNN-LSTM backbone.

### Technical Improvements:
*   **Architecture**: Modified `base/pipeline.py` to support **Split Learning Rates** automatically (detects ResNet/VGG backbones and applies 0.1x LR ratio).
*   **Regularization**: Added **Label Smoothing (0.1)** to the loss function to prevent over-confidence.
*   **Augmentation**: Switched from static `Resize` to **`RandomResizedCrop`** in the base training pipeline.
*   **Configuration**: Increased standard **Weight Decay** and **Rotation** (15 deg) settings.

## [2026-05-08] - Final Optimization (Frozen BN & Extended Patience)

### Analysis of Run: `resnet18_multiclass_20260508_191234`
*   **Result**: Peak stability reached. **Best Macro-F1: 0.6587**, **Best Val Accuracy: 68.5%**.
*   **Success**: **Frozen Batch Normalization** was highly effective at keeping the validation loss in a healthy range (0.8 - 0.9) throughout the fine-tuning phase, preventing the drift seen in earlier experiments.
*   **Success**: The extended patience (15 epochs) and higher fine-tuning LR (0.00008) allowed the model to recover quickly from the "unfreeze shock" and reach its global minimum.
*   **Verdict**: The ResNet-18 pipeline is now robust and production-ready. Further gains would likely require increased video diversity or temporal modeling.

## [2026-05-08] - Stability & Warmup Implementation
*   **Result**: High stability, but slightly lower peak performance. **Best Macro-F1: 0.6591**, **Accuracy: 69%**.
*   **Success**: The **Backbone Freezing** (5-epoch warmup) worked perfectly. Validation loss stayed controlled (0.8-1.1 range) and eliminated the wild spikes seen in previous runs.
*   **Observation**: A clear "unfreeze shock" occurred at Epoch 6, where performance dipped before recovering. This suggests the transition to full fine-tuning is the most critical phase.
*   **Issue**: Overfitting persists (94% train vs 65% val). The model reaches a plateau, and the current `lr_patience` might be triggering reductions too early.
*   **Plan**: 
    *   Implement **Frozen Batch Normalization** (keeping BN layers in eval mode during fine-tuning) to preserve ImageNet statistics.
    *   Increase `patience` to `15` and `lr_patience` to `7` to allow for longer recovery after unfreezing.
    *   Experiment with a slightly higher fine-tuning LR (`0.00008`) to escape the unfreeze dip faster.

## [2026-05-08] - Performance Optimization & Bug Fixes
*   **Result**: Significant improvement in performance. **Best Macro-F1: 0.6862**, **Accuracy: 71.17%**.
*   **Success**: The fix for the Attention module bug and the transition to a larger 80-20 validation split (leakage-safe) successfully stabilized the model and doubled the F1 score.
*   **Issue (Overfitting)**: The model still shows a heavy gap between training (96%) and validation (68%). Validation loss remains unstable and fluctuates.
*   **Identification**: Discovered that advanced augmentations in `data_config.yaml` are currently being ignored by the frame-based pipeline.
*   **Plan**: 
    *   Link `get_transforms` to the global `data_config.yaml` augmentation settings.
    *   Implement "Backbone Freezing" for the initial training phase.
    *   Refine LR (`0.00005`) and Weight Decay (`0.0001`) for better regularization.

## [2026-05-08] - Initial Pipeline Stabilization

### Analysis of Run: `resnet18_multiclass_20260508_175036`
*   **Result**: Poor performance. **Macro-F1: 0.32**. Extreme overfitting.
*   **Issue (AttributeError)**: Script crashed due to missing `frames_per_video` in config after strict enforcement was implemented.
*   **Issue (Leakage)**: The previous split logic resulted in only 1 video in the validation set, leading to wild instability and over-optimistic/random metrics.
*   **Issue (Attention Bug)**: Discovered that `SpatialAttention` module was mathematically impossible to execute for ResNet18 dimensions (512 channels) due to a hardcoded square-root check.
*   **Fixes**: 
    *   Added `frames_per_video` parameter to `hparams.yaml`.
    *   Integrated the training pipeline with the `prepare_data` manifest for leakage-safe splitting.
    *   Updated default split to 80-20 for better evaluation diversity.
    *   Refactored `SpatialAttention` to `ChannelAttention` and moved its application to before global average pooling.
