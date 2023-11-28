# Pet Monitor A.I.

This is the A.I. component of the Pet Monitor project. It is responsible for describing, training and running the model that will be used to classify the pet's image.

## Features

- Convolucional Neural Network (CNN) model for classifying the pet's image.
- Scripts for describing, training and running the model.

## Requirements

- [Python 3.8](https://www.python.org/downloads/release/python-380/)
- [TensorFlow 2.15.1](https://www.tensorflow.org/install)
- [Graphviz](https://graphviz.gitlab.io/download/)

## Setup

1. Ensure that you have Python 3.8 installed.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script to train the model:
    ```bash
    python src/train_model.py
    ```

## Dataset

The dataset contains 278 files belonging to 5 classes at 800x600 resolution of pets (different individuals, not breed), with the following classes:

- batima
- botinha
- linco
- nico
- peto

### Structure

The dataset is structured as follows:

```
data
├── batima
│   ├── 8fb31ed1- ... -077072f69f43.jpg
│   ├── 1519a832- ... -3009f8d9f4d3.jpg
│   ├── ...
│   ├── 59c7b25a- ... -00bf92a07ee5.jpg
│   ...
└── peto
│   ├── 2f68371e- ... -3ba051885d7e.jpg
│   ├── 37b3c551- ... -5db9bb3d6d34.jpg
│   ├── ...
│   └── 71d5d8bf- ... -ff89afc8374a.jpg
```
