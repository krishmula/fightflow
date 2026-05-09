# VGG-16 with transfer learning

Punch classification with **torchvision VGG-16**, ImageNet pretrained weights, and an optional spatial attention module on the 4096-D classifier bottleneck (reshaped as 64×8×8 feature maps).

## Layout

| File           | Purpose |
|----------------|---------|
| `pipeline.py`  | `VGG16PunchClassifier`, train / test / validate entrypoints |
| `hparams.yaml` | Defaults for gateway runs |
| `api.py`       | CLI glue via `BaseModelAPI` |

VGG expects **224×224** RGB normalized like other ImageNet-backed models (`base/data_config.yaml`).

## Run (project gateway)

All settings are read from `hparams.yaml`; CLI overrides are ignored.

Prepare samples:

```bash
python main.py --model vgg_16 --task prepare_data
```

Train:

```bash
python main.py --model vgg_16 --task train
```

Test / validate after replacing `REPLACE_WITH_RUN` in `hparams.yaml` with your run folder name:

```bash
python main.py --model vgg_16 --task test
```

### Data Leakage Fix (May 8, 2026)
- **Critical Issue Identified**: Frame-based models were vulnerable to data leakage because frames from the same video could appear in both training and evaluation splits. This inflated metrics and hid overfitting.
- **Solution Implemented**: Added video-level splitting using Group-aware Stratified Greedy Split (GSGS) algorithm, identical to CNN+LSTM. All frames from a video now stay in one split (train/val/test).
- **Impact**: Prevents leakage while maintaining class balance and sample diversity. Models now use leakage-safe splits by default.

### Data Preparation Optimization (May 8, 2026)
- **Performance Issue**: Extraction was slow due to re-opening video files for every frame and random seeking.
- **Optimization**: Implemented grouped video processing. The script now opens each video once and extracts all required frames sequentially in a single pass.
- **Impact**: Reduced data preparation time from minutes to seconds, significantly accelerating the development workflow.
