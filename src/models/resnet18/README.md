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

### Latest Results (May 1, 2026)

**Training Run**: `resnet18_multiclass_20260501_113142`
- **Best Epoch**: 17/27
- **Best Validation Macro F1**: 0.9332
- **Test Macro F1**: 0.8979
- **Test Accuracy**: 90.0%

#### Class-wise Performance (Test Set):
- **Straight**: F1 = 0.875 (Precision: 0.84, Recall: 0.913)
- **Hook**: F1 = 0.800 (Precision: 0.889, Recall: 0.727)
- **Uppercut**: F1 = 0.917 (Precision: 0.88, Recall: 0.957)
- **None**: F1 = 1.000 (Precision: 1.0, Recall: 1.0)

#### Key Achievements:
- **Superior Performance**: Achieved 0.933 F1 on validation, significantly outperforming the custom CNN models
- **Efficient Training**: Reached peak performance by epoch 17, demonstrating fast convergence with transfer learning
- **Balanced Classification**: Strong performance across all punch types, with perfect classification of "none" class
- **Transfer Learning Success**: Pretrained ImageNet weights provided excellent feature extraction for the limited dataset

#### Comparison with Other Models:
- **ResNet-18**: 0.898 F1 (test) - **BEST PERFORMANCE**
- **Advanced CNN**: 0.776 F1 (test)
- **Baseline CNN**: ~0.70 F1 (test)

The ResNet-18 model with transfer learning has proven to be the most effective approach, achieving 90% accuracy and 0.90 macro F1 on the test set. The model's ability to leverage pretrained features from ImageNet makes it particularly well-suited for this computer vision task with limited training data.