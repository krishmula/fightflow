# Welcome to ResNet-18 with Transfer Learning

This model uses **ResNet-18** pretrained on ImageNet with transfer learning for punch classification. ResNet-18 is a proven convolutional neural network architecture that uses residual connections to enable deeper networks.

## Why ResNet-18?

ResNet-18 was chosen for transfer learning because:
- **Pretrained on ImageNet**: Trained on 1.2 million images across 1000 classes
- **Proven Architecture**: Skip connections prevent vanishing gradients in deep networks
- **Right Size**: 18 layers provide good capacity without being too large
- **Transfer Learning**: Leverages features learned from massive datasets

## How to get started

This model follows the same structure as other models in the project, relying on the shared `base` folder infrastructure.

Key files in this folder:

1. **`pipeline.py`**: Contains the `ResNet18PunchClassifier` class that loads pretrained ResNet-18 and replaces the final layer for 4-class punch classification.
2. **`hparams.yaml`**: Transfer learning focused hyperparameters with smaller learning rates and appropriate batch sizes.
3. **`api.py`**: Model registration and configuration interface.

## Transfer Learning Strategy

The model uses a two-phase training approach:
1. **Phase 1**: Freeze the pretrained backbone, train only the final classifier layer
2. **Phase 2**: Unfreeze the backbone for fine-tuning with smaller learning rates

## Making changes

* **I want to change the neural network architecture:** Edit the `ResNet18PunchClassifier` class in `pipeline.py`.
* **I want to adjust transfer learning settings:** Modify the learning rate and layer freezing in `hparams.yaml`.
* **I want to change data augmentations:** Edit `src/models/base/data_config.yaml`.
* **I want to change training infrastructure:** Edit `src/models/base/pipeline.py`.

## How to Run

Use the main project CLI script for all operations.

**To train the ResNet-18 model:**
```bash
python main.py --model resnet18 --task train
```

**To evaluate the trained model:**
```bash
python main.py --model resnet18 --task test
```

## Model Improvements Timeline

### Initial Implementation (May 1, 2026)
- ResNet-18 with ImageNet pretrained weights
- Custom 4-class classifier head with dropout
- Transfer learning focused hyperparameters
- Smaller learning rate (0.0001) for fine-tuning
- **Code Cleanup**: Removed unused parameters and standalone script execution
- Goal: Push F1 score above 0.90 with transfer learning advantage

### Enhanced Data Augmentations (May 1, 2026)
- **Motion-specific transforms**: Random rotations (±15°) and affine transforms for motion variation
- **Random erasing**: Cutout-style augmentation (30% probability) for robustness
- **Improved data pipeline**: Real-time augmentation during training, no preprocessing needed
- **Purpose**: Better generalization to different punch execution styles

### Spatial Attention Module (May 1, 2026)
- **Channel attention mechanism**: Focuses on motion-relevant features in final layer
- **Lightweight design**: Only 32,768 additional parameters (0.28% increase)
- **Transfer learning friendly**: Learns quickly with pretrained weights
- **Purpose**: Improved focus on punch motion patterns, especially for hook detection

### Latest Results (May 4, 2026)

**Training Run**: `resnet18_multiclass_20260504_223017`
- **Best Epoch**: 30/30
- **Best Validation Macro F1**: 0.861
- **Validation Accuracy**: 0.860
- **Dataset**: 1,464 train samples, 314 val samples (balanced to 590 per class)
- **Training Config**: 30 epochs, LR 0.0001→5e-05, batch size 16, MPS device

#### Class-wise Performance (Validation Set):
- **Straight**: F1 = 0.823 (Precision: 0.813, Recall: 0.833)
- **Hook**: F1 = 0.814 (Precision: 0.894, Recall: 0.747)
- **Uppercut**: F1 = 0.849 (Precision: 0.792, Recall: 0.916)
- **None**: F1 = 0.959 (Precision: 0.972, Recall: 0.946)

#### Key Achievements:
- **Solid Performance**: Achieved 0.861 F1 on validation set with transfer learning
- **Strong Generalization**: "None" class near-perfect (0.959 F1), "Uppercut" high recall (0.916)
- **Hook Challenge**: Lower recall (0.747) suggests need for more hook-specific features
- **Stable Training**: Converged steadily with LR decay, minimal overfitting
- **Efficient**: Good results with moderate dataset size and compute

#### Comparison with Other Models:
- **ResNet-18**: 0.861 F1 (val) - Strong baseline with transfer learning
- **Advanced CNN**: 0.776 F1 (val)
- **Baseline CNN**: ~0.70 F1 (val)

The ResNet-18 model demonstrates reliable performance with transfer learning, though there's room for improvement on the "hook" class through enhanced data augmentation or attention mechanisms.