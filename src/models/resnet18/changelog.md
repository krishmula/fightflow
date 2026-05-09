# ResNet-18 Model Changelog

All notable changes, experimental results, and optimization notes for the ResNet-18 punch classifier are documented here.

## [2026-05-08] - Performance Optimization & Bug Fixes

### Analysis of Run: `resnet18_multiclass_20260508_182538`
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
