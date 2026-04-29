# Baseline CNN Training and Testing Pipeline

This baseline keeps all dataset assets under the existing `data/` folder.

## Model API Structure

Model code is organized per model folder, so each model can expose an API-like surface:

```text
models/
  baseline_cnn/
    api.py
    pipeline.py
    inference.py
```

The gateway calls a selected model + task directly:

```text
main.py
```

## Folder layout

Place media files (videos or images) in this structure:

```text
data/classification/boxing4cls/
  train/
    jab/
    hook/
    uppercut/
    negative/
  val/
    jab/
    hook/
    uppercut/
    negative/
  test/
    jab/
    hook/
    uppercut/
    negative/
```

Notes:
- Every split should contain all 4 class folders.
- You can place full videos directly in class folders; training/testing will sample frames internally.
- If `val/` has no media, the script auto-splits a validation subset from `train/`.
- Test evaluation is skipped during training if `test/` has no media.

## Train

Run from project root:

```bash
.venv/bin/python main.py --model baseline_cnn --task train
```

Useful options:

```bash
.venv/bin/python main.py --model baseline_cnn --task train -- \
  --epochs 25 \
  --batch-size 32 \
  --image-size 128 \
  --lr 1e-3 \
  --use-class-weights \
  --device auto
```

## YAML Hyperparameters

The model's settings, including paths and hyperparameters, are strictly controlled by `models/baseline_cnn/hparams.yaml`.

You can modify settings like `epochs`, `batch_size`, or `learning_rate` directly in this file.

### Binary Setup: Jab vs Not Jab

To train a binary classifier, change the `class_names` under the `train:` section in `hparams.yaml` to:

```yaml
class_names: jab,not_jab
```

`not_jab` is handled automatically as all non-jab folders in each split
(`hook`, `uppercut`, `negative`, etc.), plus an optional explicit `not_jab/` folder.

## Test

Evaluates the checkpoint defined under `test:` in `hparams.yaml`:

```bash
.venv/bin/python main.py --model baseline_cnn --task test
```

## Validate

Evaluates the checkpoint defined under `validate:` in `hparams.yaml`:

```bash
.venv/bin/python main.py --model baseline_cnn --task validate
```

## Video Inference

Runs visual inference over a raw video (paths defined under `inference:` in `hparams.yaml`) and renders bounding boxes/labels:

```bash
.venv/bin/python main.py --model baseline_cnn --task inference
```

## Outputs

Training outputs are saved per run under:

```text
models/baseline_cnn/runs/<run_name_timestamp>/
  checkpoints/
    best.pt
    last.pt
  reports/
    val_metrics.json
    val_classification_report.json
    val_confusion_matrix.csv
    test_metrics.json
    test_classification_report.json
    test_confusion_matrix.csv
  config.json
  history.csv
  run_summary.json
```

Test-only outputs are saved under:

```text
models/baseline_cnn/eval/<test_run_name>/
  test_metrics.json
  test_classification_report.json
  test_confusion_matrix.csv
  test_run_details.json
```
