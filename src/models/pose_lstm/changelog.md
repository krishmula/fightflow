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

## [2026-05-12] - Optimization & Feature Engineering Session

### Phase 1: Robustness & Reporting
- **Safe Evaluation**: Fixed `ValueError` crash on empty test splits by adding dataset length guards in `run_epoch` and `evaluate_and_save`.
- **Reporting Alignment**: Standardized output with `cnn_lstm` pattern, including printed JSON summaries and `run_summary.json` generation.


### Phase 2: Overfitting & Generalization Fixes
- **Architecture Downsizing**: Reduced model to 1 layer / 64 hidden units to prevent parameter-based memorization on the small (21-video) dataset.
- **Stronger Regularization**: Increased `weight_decay` to 0.2 and `pose_noise_std` to 0.05.
- **Result**: Successfully closed the training-validation gap from ~25% to ~18%.

### Phase 3: Motion & Geometry Engineering
- **Velocity Features**: Added 99 temporal difference features (frame-to-frame joint movement) to the input vector.
- **Temporal Attention**: Replaced global average pooling with a learnable attention layer to focus on the "impact" frames of a punch.
- **Horizontal Scaling Augmentation**: Added random X-axis scaling (0.9x-1.1x) to simulate varied reach lengths.
- **Result**: Reduced "None vs. Hook" confusion by 35% and stabilized Macro-F1 at 0.62 (Run: `220145`).

### Phase 4: Vector Directionality (Current)
- **Relative 3D Vectors**: Replaced scalar Euclidean distances with 18 raw 3D offsets (dx, dy, dz) between wrists and (Shoulder, Hip, Nose).
- **Goal**: Restore Uppercut accuracy by preserving the vertical (dy) vs. horizontal (dx) extension components.

### Phase 5: DRY Codebase & Lateral Kinematics (Current)
- **Centralized Utilities**: Moved duplicated codebase operations (loading config, set_seed, checkpoints, data splits) into `src/models/base/utils.py`.
- **Hip-Centering Normalization**: Subtracted the mid-hip coordinate from all landmarks to remove absolute screen-space variance.
- **Max Absolute Velocity Features**: Extracted maximum X, Y, Z velocities per clip for both wrists to explicitly capture the lateral-heavy "swing" of a hook vs. the forward "push" of a straight.
- **Model Tuning**: Restored LSTM capacity (128 hidden, 2 layers) and dropped aggressive weight decay (to 0.01) to process the new 228-dimension feature vector without underfitting.
- **Result**: Accuracy spiked from 59.8% to 69.6% (Macro-F1 0.663). Uppercut F1 hit 0.80. Hook F1 improved but remains the primary bottleneck due to temporal overlap constraints.

### Phase 6: Explicit Joint Angles
- **Feature Addition**: Added the 3D Elbow Angle (angle between Shoulder, Elbow, and Wrist) for both arms as explicit features, bringing `input_dim` to 230.
- **Goal**: Help the model distinguish between a Straight (arm extends to ~180°) and a Hook (arm stays bent at ~90°) since the raw forward/lateral velocities weren't sufficient alone.
- **Result**: Hook Recall skyrocketed from 32% to 54% (F1 jumped to 0.43). Uppercut F1 hit an all-time high of 0.82. However, overall Macro-F1 dipped slightly to 0.651 because the model began over-relying on the bent elbow feature, misclassifying some "None" guard positions and sloppy Straights as Hooks. This solved the Hook blindspot, trading a bit of precision for massive recall gains.

### Phase 7: Relative Velocity & Class Weight Tuning
- **Disabled Class Weights**: Switched `use_class_weights` to `false`. Observed that statistical weighting had little impact, confirming the model is feature-driven rather than frequency-driven.
- **Wrist-to-Face Velocity**: Added the 3D velocity of the vector between the Nose and each Wrist (6 dimensions).
- **Goal**: Isolate arm punch explosiveness from whole-body movement (e.g. bobbing and weaving).
- **Result**: Massive success. **Macro-F1 jumped to 0.683** and **Accuracy reached 69.1%**. This successfully decoupled the "bent elbow" feature from the "explosive punch" feature, reducing `None` vs `Hook` false positives and pushing Uppercut F1 to 0.84.

### Phase 8: Data Augmentation Experiments (Current)
- **Horizontal Flip (FAILED)**: Implemented 50% chance of X-axis mirroring + Left/Right index swapping. 
    - **Result**: Tanked performance back down to 0.62 Macro-F1. The model was unable to resolve the biomechanical asymmetries of Uppercuts when mirrored, leading to "Frankenstein" pose confusion. **Reverted immediately.**
- **Temporal Shift Augmentation**: Implemented random +/- 2 frame temporal shifting with edge-padding during training to simulate varied punch timing.
- **Result**: Stabilized training. Pushed **overall Accuracy to a record high of 69.2%** and significantly improved Straight F1 (0.62). Confirmed that temporal variety is safe for boxing biomechanics while spatial mirroring is not.
