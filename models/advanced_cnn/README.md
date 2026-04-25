# Advanced Residual CNN

This model represents the second evolution of the boxing action classifier. It improves upon the baseline by introducing architectural patterns used in modern high-performance networks.

## Improvements & Rationale

### 1. Residual Connections (Skip-Connections)
- **What:** We added `ResidualBlock` modules that allow the input to bypass certain layers and be added directly to the output of the block.
- **Why:** In the baseline model, deep layers often lose low-level spatial information. Residual connections help gradients flow better during training and allow the model to preserve "raw" features while learning abstract ones.

### 2. Larger Initial Receptive Field (5x5 Kernels)
- **What:** The first layer was upgraded from a `3x3` kernel to a `5x5` kernel with a stride of 2.
- **Why:** Boxing actions (like jabs) involve large body movements across the frame. A larger initial kernel helps the model capture the "big picture" (the whole boxer) before focusing on small details in deeper layers.

### 3. Increased Feature Capacity
- **What:** We increased the maximum channel depth from 256 to 512.
- **Why:** To identify complex movements, the model needs more "filters" to distinguish between a jab, a step forward, or a simple hand adjustment.

### 4. Advanced Classifier Head
- **What:** Instead of going straight to the output, we added a hidden dense layer with 256 neurons and increased Dropout (0.4).
- **Why:** This creates a "decision layer" that integrates all visual features before making a final classification. The higher dropout specifically targets the overfitting issue observed in the baseline model.

## Performance & Results

After a full training run of 30 epochs (with early stopping at Epoch 21), this model showed a massive improvement over the Baseline CNN.

### Benchmark Comparison

| Metric | Baseline CNN | **Advanced CNN** | Improvement |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | 87.4% | **89.5%** | +2.1% |
| **Test Accuracy** | 64.7% | **84.2%** | **+19.5%** |
| **Test Macro F1** | 0.647 | **0.839** | **+0.192** |

### Key Takeaway
The "Generalization Gap" (the difference between validation and test accuracy) dropped from **22.7% to 5.3%**. This confirms that the **Residual connections** and **higher Dropout (0.4)** successfully mitigated overfitting, allowing the model to perform reliably on entirely new video clips.

## Usage
To train this model:
```bash
python main.py --model advanced_cnn --task train
```
