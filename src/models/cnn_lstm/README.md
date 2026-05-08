# CNN+LSTM (Sample Classification)

This model classifies boxing techniques from short RGB samples. Each sample is centered on the annotated frame (full arm extension), and the CNN backbone extracts per-frame features that an LSTM aggregates across time.

## Data preparation

- Frame-only CNNs extract single-frame samples into `data/processed/frames`.
- CNN+LSTM extracts fixed-length clip samples into `data/processed/clips` and writes `data/processed/clip_manifest.csv`.

The sample window is controlled by `clip_pre_frames` and `clip_post_frames` in [src/models/base/data_config.yaml](../base/data_config.yaml). The loader will pad or subsample to `clip_length` (in [src/models/cnn_lstm/hparams.yaml](hparams.yaml)) at training time.

Prepare samples:

```bash
python main.py --model cnn_lstm --task prepare_data
```

## Backbones you can plug in

`cnn_baseline`, `cnn_advanced`, `resnet18`, `vgg_16`. Set `cnn_backbone`, `cnn_init`, and `freeze_backbone` in [src/models/cnn_lstm/hparams.yaml](hparams.yaml).

## Training

All settings are read from `hparams.yaml`; CLI overrides are ignored.

```bash
python main.py --model cnn_lstm --task train
```

## Data splits

Splits are leakage-safe by grouping on `video_id`, so all samples from a video stay in the same bucket. This prevents the same source video from appearing in both training and evaluation, which would inflate metrics and hide overfitting.

The splitter then assigns whole videos to each split with two goals:
1) Match the target train/val/test ratios as closely as possible by total sample count.
2) Preserve per-class balance by tracking label counts contributed by each video.

Implementation summary (heuristic algorithm):
- **Algorithm name:** Group-aware Stratified Greedy Split (GSGS).
- **In simple terms:** treat each video as a single item and place it into train or val so that the total number of samples and the class mix stay close to the target ratios, while never splitting a video across buckets.
- This is a greedy, leakage-safe bin-packing heuristic with multi-objective cost (ratio matching + per-class balance).
- Compute per-video label histograms and total sample counts.
- Compute target totals for each split (overall samples and per-class samples) from the ratios.
- Sort videos by size (largest first) and greedily assign each video to the split that minimizes deviation from the targets.

This keeps leakage out while still keeping validation small and class distributions stable, which is especially important for small datasets.

## Evaluation

```bash
python main.py --model cnn_lstm --task validate
python main.py --model cnn_lstm --task test
```

## Key settings (hparams)

- `clip_length`: number of frames per sample used by the LSTM
- `cnn_backbone`: which CNN to use for per-frame features
- `cnn_init`: weight init source (latest, imagenet, random, or a checkpoint path)
- `freeze_backbone`: whether to freeze the CNN during LSTM training
- `lstm_hidden`, `lstm_layers`, `lstm_bidirectional`, `lstm_output`: LSTM configuration
- `image_size`: input size for the CNN (224 for resnet18/vgg_16)

Defaults live in [src/models/cnn_lstm/hparams.yaml](hparams.yaml).
