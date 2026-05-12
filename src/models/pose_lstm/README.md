# Pose-based LSTM classifier for Boxing Moves

This model uses MediaPipe Pose skeletons extracted from video clips to classify boxing moves (straight, hook, uppercut, or none).

## Key Features
- **Background Agnostic**: Uses 3D human pose coordinates instead of raw pixels.
- **Fast Training**: Operates on tiny vector sequences (~100 numbers per frame) instead of heavy images.
- **Robust Motion Capture**: Captures body mechanics and joint trajectories.

## Folder Structure
- `api.py`: Standard API adapter for `main.py`.
- `pipeline.py`: Training, validation, and testing logic.
- `utils.py`: Pose-specific dataset and training helpers.
- `hparams.yaml`: Model and training configuration.
- `changelog.md`: Record of experimental runs and performance.

## Usage
### 1. Data Preparation
Extract poses from existing clips:
```bash
python3 main.py --model pose_lstm --task prepare_data
```

### 2. Training
```bash
python3 main.py --model pose_lstm --task train
```

### 3. Evaluation
```bash
python3 main.py --model pose_lstm --task validate
python3 main.py --model pose_lstm --task test
```
