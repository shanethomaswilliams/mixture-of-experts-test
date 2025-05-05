# Data Directory

This folder contains datasets required for training and evaluating the models. Below are instructions on how to populate this folder.

## Downloading CIFAR-10 Dataset

To download the CIFAR-10 dataset, use the `src/download-cifar10.py` script. Run the following command:

```bash
python src/download-cifar10.py
```

This will download the CIFAR-10 dataset and save it in the `data/` folder.

## Folder Contents

After downloading, this folder will contain:

- `cifar-10-batches-py/`: The CIFAR-10 dataset in Python format.

Ensure that the dataset is correctly downloaded before proceeding with training or experiments.