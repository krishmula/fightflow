# Welcome to the Base Infrastructure!

If you are a new developer starting on this project, welcome! This folder is the heart of the modeling pipeline. 

To make your life easier, we have separated the complex boilerplate code (like loading data, running training loops, evaluating metrics, and creating graphs) from the actual neural network designs. This entire `base` folder handles the heavy lifting so you do not have to.

## How the Architecture Works

When you look at other folders like `cnn_baseline` or `cnn_advanced`, you will notice they are very small. That is by design! They only contain PyTorch classes defining the shapes of the neural networks. When someone wants to train `cnn_baseline`, it automatically calls the files in this `base` folder to do the actual work.

## What is inside this folder?

1. **`pipeline.py`**: This is the engine. It contains the shared `train`, `test_only`, and `validate_only` functions. It reads your dataset, handles PyTorch DataLoaders, calculates the loss, plots your training history, and saves the best model checkpoints.
2. **`api.py`**: This file acts as the gateway. It parses command line arguments and includes a `prepare_data` script that either extracts single-frame samples (frame-only CNNs) or clip samples (CNN+LSTM), depending on the selected model.
3. **`data_config.yaml`**: This is the master configuration file for your dataset. If you want to automatically balance your classes to prevent bias, or if you want to tweak image augmentations like brightness or random flips, you do it here.
4. **`utils.py`**: Small helper tools, like automatically detecting if you are using a GPU or CPU.

## How to create a new model

If you are tasked with creating a brand new model (for example, an LSTM or a Vision Transformer), you will follow these simple steps:

1. Create a new folder (e.g., `my_new_model`).
2. Inside that folder, create a `pipeline.py` file. Write your standard PyTorch `nn.Module` class here.
3. Import `train`, `test_only`, and `validate_only` from `base.pipeline`. Create tiny functions that pass your new PyTorch class into those shared functions.
4. Create an `api.py` file that inherits from `BaseModelAPI` to handle your default configurations.
5. Create a `hparams.yaml` file to set your default learning rates and batch sizes.

You do not need to write a custom training loop! You just write the math, and this `base` folder takes care of the pipeline.

## How to Run

Because this `base` folder is purely infrastructure, you do not run it directly. Instead, you run the models that rely on it using the main project gateway (`main.py`).

For example, the data preparation script lives in `base` but now uses the model flag to choose sample type (single-frame vs clip):
```bash
python main.py --model <model_name> --task prepare_data
```

## Data Splits and Leakage

This project uses video-level splitting for training/validation/testing to avoid data leakage. A sample-level split can place clips or frames from the same video into multiple splits, which inflates performance and can look like overfitting because the model learns video-specific cues. Video-level splits keep all samples from a video together, which is safer but can lead to uneven sample counts across splits.

