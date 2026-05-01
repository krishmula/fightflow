# Welcome to the Advanced CNN

If you are a new developer, welcome to the advanced model folder! This is our flagship Convolutional Neural Network. It is designed to be much more powerful than the baseline model.

We achieve higher accuracy here by using a deeper network structure and a technique called residual connections (skip connections). 

## How to get started

Just like all other models in this project, this folder relies entirely on the `base` folder to handle data loading, training loops, and metric evaluation. This allows you to focus purely on the deep learning architecture.

Here is a quick tour of the files in this folder:

1. **`pipeline.py`**: This is the file you care about the most. It defines two PyTorch classes:
   * `ResidualBlock`: A custom layer that allows data to skip past convolutions. This prevents the network from forgetting the original image as the data moves deeper into the model.
   * `AdvancedCNN`: The main network. It starts with a large kernel to scan the whole image, then stacks multiple `ResidualBlock` layers, and finishes with a robust classifier.
2. **`hparams.yaml`**: This configuration file contains the training settings. Because this model is deeper than the baseline, you will likely find different learning rates or patience settings here. You can change these numbers to experiment with training speeds.
3. **`api.py`**: This file acts as the bridge connecting this network to our command-line tools. You do not need to edit this file.

## Making changes

* **I want to tweak the neural network design:** Edit the `AdvancedCNN` or `ResidualBlock` classes inside `pipeline.py`.
* **I want to change the learning rate or batch size:** Edit the numbers inside `hparams.yaml`.
* **I want to change image augmentations (like flips):** Do not edit this folder! Go to `src/models/base/data_config.yaml`.
* **I want to change the loss function or early stopping:** Do not edit this folder! Go to `src/models/base/pipeline.py`.

By keeping the heavy lifting in the `base` folder, you are completely free to break, rebuild, and experiment with the PyTorch math in `pipeline.py` without risking the stability of the rest of the application!

## How to Run

To run the advanced model, you will use the main project CLI script located in your root directory.

**To extract video frames into uniform classes (this is universal and does not require a model flag):**
```bash
python main.py --task prepare_data
```

**To train the advanced model:**
```bash
python main.py --model cnn_advanced --task train
```

**To evaluate the trained model on your test set:**
```bash
python main.py --model cnn_advanced --task test
```

## Model Improvements Timeline

### Initial Architecture (April 2026)
- Basic residual CNN with 2 residual blocks
- 128x128 input resolution
- Simple classifier head
- Binary classification achieved 0.89 F1 score
- Multiclass struggled with uppercut detection (poor performance)

### Architecture Enhancements (April 30, 2026)
- Added attention mechanism for better feature focusing
- Increased depth to 3 residual blocks
- Enhanced classifier with batch normalization and dropout
- Multiclass performance improved to 0.73 F1 score

### Major Improvements (May 1, 2026)
- Increased input resolution to 224x224 for better detail
- Added deeper feature extraction (1024 channels)
- Implemented stronger regularization (dropout 0.5, batch norm)
- Fixed data splits (90% train, 5% val, 5% test)
- Best multiclass performance: 0.88 F1 score with 77.8% test accuracy
- Uppercut detection improved from 0.63 to 0.80 F1 score
- Hook detection improved from 0.56 to 0.67 F1 score
