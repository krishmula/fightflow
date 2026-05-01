# CNN Training and Testing Pipeline

This model trains directly on raw video frames mapped by `data/annotations.csv`.
There is no need for pre-cut folders or intermediate split directories.

## Model API Structure

Model code is organized per model folder, so each model can expose an API-like surface:

```text
models/
  [model_name]_cnn/
    api.py
    pipeline.py
    hparams.yaml
```

The gateway calls a selected model + task directly:

```text
main.py
```

## Data Loading & Splitting

The model directly reads `data/annotations.csv` and slices frames on the fly from `data/downloaded-videos/`.

Data splits are handled dynamically and can be configured natively inside `hparams.yaml` using the `splits:` block:

```yaml
train:
  class_names: straight,hook,uppercut,none
  splits:
    train: 0.70
    val: 0.15
    test: 0.15
```

Notes:
- `straight` is natively used instead of `jab`. No manual mapping is performed behind the scenes.
- Classes are mapped strictly to the strings provided in `class_names`.
- Random seed for splits is configured in `hparams.yaml` or passed via CLI argument `--seed`.

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

The model's settings, including paths and hyperparameters, are controlled by `hparams.yaml`.

You can modify settings like `epochs`, `batch_size`, or `splits` directly in this file.

### Multi-Class Setup

The model natively trains on the four standard classes found in your dataset:

```yaml
train:
  class_names: straight,hook,uppercut,none
```

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
