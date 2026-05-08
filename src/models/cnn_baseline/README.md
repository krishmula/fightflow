# Welcome to the Baseline CNN

If you are new to the project, this folder is the perfect place to start. This is our baseline Convolutional Neural Network (CNN). It is the simplest and fastest model we have. 

We use this model to check if our dataset is working correctly and to provide a basic score to compare against our more complex models.

## How to get started

You might be surprised by how few lines of code are in this folder. That is because we use a shared infrastructure located in the `base` folder. 

Here is what you need to know about the files in this folder:

1. **`pipeline.py`**: This is where the actual math happens. If you open this file, you will see a class called `BaselineCNN`. This is standard PyTorch code. If you want to experiment by adding another layer, changing the dropout percentage, or swapping the activation functions, this is the only file you need to touch!
2. **`hparams.yaml`**: This file contains the training settings specifically for this baseline model. You can open this file to easily change the learning rate, the batch size, or the number of training epochs without touching any Python code.
3. **`api.py`**: You generally do not need to touch this file. It simply registers the model with the rest of our application.

## Making changes

* **I want to change the neural network shape:** Edit the `BaselineCNN` class in `pipeline.py`.
* **I want to change the learning rate:** Edit `hparams.yaml`.
* **I want to change how the dataset is loaded or balanced:** Do not edit anything in this folder! Go to `src/models/base/data_config.yaml`.
* **I want to change how the training loop saves checkpoints:** Do not edit anything in this folder! Go to `src/models/base/pipeline.py`.

This clean separation means you can safely experiment with the neural network architecture here without worrying about breaking the data loaders or metrics systems!

## How to Run

To interact with this model, use the main project CLI script located in your root folder.

All settings are read from `hparams.yaml`; CLI overrides are ignored.

**To extract samples into uniform classes:**
```bash
python main.py --model cnn_baseline --task prepare_data
```

**To train the baseline model:**
```bash
python main.py --model cnn_baseline --task train
```

**To evaluate the trained model on your test set:**
```bash
python main.py --model cnn_baseline --task test
```

## Model Improvements Timeline

### Initial Setup (April 2026)
- Basic CNN architecture with 128x128 input resolution
- Simple data augmentation (horizontal flip, color jitter)
- Binary classification achieved 0.87 F1 score

### Data Pipeline Fixes (April 30, 2026)
- Fixed path resolution bugs in data loading
- Implemented automatic class balancing (uniform_samples_per_class: "auto")
- Multiclass classification improved from poor performance to 0.72 F1 score

### Training Stability (May 1, 2026)
- Increased training epochs from 30 to 50
- Adjusted data splits to 90% train, 5% val, 5% test
- Improved early stopping patience and learning rate scheduling
- Best multiclass performance: 0.72 F1 score with 72.8% test accuracy
