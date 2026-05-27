# CNN-Sentic — Speech Emotion Recognition

CNN-Sentic is a Convolutional Neural Network (CNN)-based speech emotion recognition project that classifies emotions from short `.wav` audio clips. The pipeline extracts audio features such as MFCC, chroma, and mel-spectrogram features, trains a CNN model, saves the trained artifacts, and provides a Tkinter GUI for model loading, visualization, and speech emotion prediction.

## Table of Contents

- [Overview](#Overview)
- [Emotion Classes](#Emotion-Classes)
- [Training Results](#Training-Results)
- [Dataset](#Dataset)
- [Project Structure](#Project-Structure)
- [Setup](#Setup)
- [Usage](#Usage)
- [Expected Files](#Expected-Files)
- [Screenshots](#Screenshots)
- [Limitations](#Limitations)
- [Troubleshooting](#Troubleshooting)
- [Future Improvements](#Future-Improvements)
- [References](#References)
- [Contributors](#Contributors)

## Overview

This repository contains the components required to build and test a speech emotion recognition workflow based on audio feature extraction and CNN training. It is intended as a research or demonstration project rather than a production-ready application.

## Emotion Classes

The model is designed to classify speech into emotion categories based on the dataset labels.

Common speech emotion labels may include:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

> Note: The exact classes depend on the labels present in the downloaded dataset and the preprocessing logic used in `SpeechTrain.py`.

## Training Results

The latest saved training run produced the following results:

- **Final training accuracy:** `95.33%`
- **Final training loss:** `0.1667`

> Important: These values come from the saved training history file at `model/speechhistory.pckl`. They represent training-set performance only. They should not be interpreted as validation or test accuracy.

A separate validation/test split, confusion matrix, and classification report should be added before using this project as a fully evaluated machine learning system.

## Dataset

The datasets should be downloaded directly from Kaggle according to their dataset license and terms. The dataset files are not included in this repository because of size and licensing considerations.

This project uses the Kaggle speech emotion datasets:

- https://www.kaggle.com/datasets/rasmitha26/speechemotiondataset1
- https://www.kaggle.com/datasets/rasmitha26/speechemotiondataset2

### Recommended download method

Before running the commands below, configure Kaggle credentials using `~/.kaggle/kaggle.json` or the official Kaggle CLI authentication method. Avoid committing API keys or credentials to GitHub.

1. Install Kaggle CLI and authenticate with your Kaggle credentials.
2. Run:

```bash
kaggle datasets download -d rasmitha26/speechemotiondataset1 -p . --unzip
kaggle datasets download -d rasmitha26/speechemotiondataset2 -p . --unzip
```

3. Create the expected dataset directory and merge both extracts:

```bash
mkdir -p SpeechEmotionDataset
mv SpeechEmotionDataset1/SpeechEmotionDataset/* SpeechEmotionDataset/
mv SpeechEmotionDataset2/SpeechEmotionDataset/* SpeechEmotionDataset/
```

### Manual download

If you prefer not to use the CLI, download both zip files from Kaggle, extract them locally, and then move the extracted `SpeechEmotionDataset/Actor_*` folders into the repository root under `SpeechEmotionDataset/`.

### Expected dataset layout

```
SpeechEmotionDataset/
├── Actor_01/
├── Actor_02/
├── ...
├── Actor_24/
```

The training script expects the dataset root to be named `SpeechEmotionDataset`.

## Project Structure

```
Sentic-CNN-Audio/
├── Main.py
├── SpeechTrain.py
├── requirements.txt
├── README.md
├── .gitignore
├── testspeech/          # example audio files for quick testing
├── SpeechEmotionDataset/ # expected dataset root for training
└── model/              # trained model artifacts and saved data
```

## Setup

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

For best results, train the model first using `SpeechTrain.py`, then launch the GUI using `Main.py`.

### Train the model

Run the training script with the dataset in place:

```bash
python SpeechTrain.py
```

This script will:

- load `.wav` files from `SpeechEmotionDataset/`
- extract MFCC, chroma, and mel-spectrogram features
- build and train the CNN model
- save model files and training history in `model/`

### Run the GUI

Use the GUI to load saved models and predict speech emotion:

```bash
python Main.py
```

Use the buttons to upload datasets, preprocess data, train or load models, and predict emotion from speech recordings.

## Expected Files

After training, the `model/` directory should contain:

- `speechmodel.json`
- `speech_weights.weights.h5`
- `speechhistory.pckl`
- `speechX.txt.npy`
- `speechY.txt.npy`

## Screenshots

Add screenshots or a demo GIF here to show the GUI workflow and prediction output.

## Limitations

- The reported accuracy is training accuracy only, not validation or test accuracy.
- The current pipeline may overfit if no separate validation/test split is used.
- Prediction quality depends on dataset quality, recording conditions, background noise, and speaker variation.
- The GUI is intended for local desktop use and is not deployment-ready.
- Model artifacts are generated after training and may not exist in a fresh clone.

## Troubleshooting

- If `SpeechTrain.py` fails due to missing data, verify that `SpeechEmotionDataset/` exists and contains `.wav` files.
- If Kaggle CLI reports missing credentials, either set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables, or create `~/.kaggle/kaggle.json` with your Kaggle API token.
- If `Main.py` cannot load models, ensure the `model/` folder contains the required files.
- If kaggle is not recognized, install the Kaggle CLI with `pip install kaggle`.
- If model files are missing, run `python SpeechTrain.py` before launching the GUI.
- If audio prediction fails, verify that the input file is a valid `.wav` file and matches the expected sampling/feature extraction format.
- If dependency installation fails, check the Python version and confirm that all packages in `requirements.txt` are compatible.

## Future Improvements

- Add a proper train/validation/test split
- Report validation accuracy, test accuracy, precision, recall, F1-score, and confusion matrix
- Improve feature extraction and CNN architecture
- Add noise handling and audio normalization
- Add command-line batch inference support
- Add deployment-ready packaging for desktop or web
- Rename generated artifacts to cleaner names such as `features.npy`, `labels.npy`, and `training_history.pkl`

## References

- Kaggle dataset 1: https://www.kaggle.com/datasets/rasmitha26/speechemotiondataset1
- Kaggle dataset 2: https://www.kaggle.com/datasets/rasmitha26/speechemotiondataset2
- librosa documentation: https://librosa.org/
- TensorFlow/Keras documentation: https://www.tensorflow.org/

## Contributors

- Rasmitha Chinthalapally
