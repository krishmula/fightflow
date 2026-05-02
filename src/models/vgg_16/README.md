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

Train:

```bash
python main.py --model vgg_16 --task train -- --config src/models/vgg_16/hparams.yaml
```

Test / validate after replacing `REPLACE_WITH_RUN` in `hparams.yaml` with your run folder name:

```bash
python main.py --model vgg_16 --task test -- ...
```
