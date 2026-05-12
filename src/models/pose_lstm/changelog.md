# Pose-LSTM Training Changelog

## [2026-05-12] - Initial Skeleton-based Pipeline

### Summary of Run: `pose_lstm_initial_baseline`
*   **Method**: MediaPipe Pose Extraction + Bi-LSTM.
*   **Technique**: Standard 11-frame window, 2-layer LSTM (128 hidden), dropout 0.5.
*   **Observation**:
    *   Transitioned from pixel-based CNN-LSTM to vector-based Pose-LSTM.
    *   Integrated into the central `BaseModelAPI` for standardized preparation.

## [2026-05-12] - High Performance Baseline
### Summary of Run: `pose_lstm_baseline_v2`
*   **Result**: Best validation Macro-F1: **0.8727** at epoch 49.
*   **Fixes**: Added `zero_division=0` to evaluation metrics to handle missing classes in small validation/test splits.
*   **Observation**: Pose-based features are showing significantly higher baseline performance and better stability compared to the raw pixel CNN-LSTM (which previously hit ~0.70-0.75).

## [2026-05-12] - Fine-Tuning & Visualization Suite
- **Added `resume` support**: The training pipeline can now load a checkpoint to start fine-tuning or continue training.
- **Implemented `overlay` task**: New task to generate videos with MediaPipe skeleton overlays and model predictions.
- **Output Organization**: Overlay videos are now automatically saved in the `inference/` folder of the associated run.
- **Improved Configurability**: Moved hardcoded MediaPipe task paths and confidence thresholds to `hparams.yaml`.
- **API Cleanup**: Inherited `prepare_data` from base API for better consistency across models.

### 2026-05-12 - Advanced Regularization & Augmentation
- **Implemented Label Smoothing**: Added `label_smoothing` (0.1) to `nn.CrossEntropyLoss` to prevent overfitting and improve generalization.
- **Added Pose Jitter Augmentation**: Updated `PoseDataset` to inject Gaussian noise (`pose_noise_std: 0.01`) into training coordinates.
- **Increased Weight Decay**: Pushed `weight_decay` to `0.1` for stronger regularization during fine-tuning.
