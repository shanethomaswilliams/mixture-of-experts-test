# Mixture of Experts Test Repository

This repository contains code for training and evaluating models using a mixture of experts approach. Below is a detailed explanation of the repository structure, the models implemented in the `src/models.py` file, and instructions on how to run experiments.

## Repository Structure

- `data/`: Contains datasets and related files. See `data/README.md` for details on how to populate this folder.
- `experiments/`: Stores experiment configurations and results. See `experiments/README.md` for details on running experiments.
- `notebooks/`: Contains Jupyter notebooks for analysis and visualization.
- `run/`: Scripts for running experiments and generating plots.
- `src/`: Source code for models, utilities, and dataset handling.

## Models in `src/models.py`

The `src/models.py` file implements various models used in the mixture of experts approach. These include:

- **Expert Models**: Individual models that specialize in specific tasks or subsets of data.
- **Gating Mechanism**: A model that dynamically routes inputs to the appropriate expert(s).
- **Ensemble Models**: Combines outputs from multiple experts for final predictions.

Refer to the comments in `src/models.py` for detailed documentation on each model.

## Running Experiments

### Training Models

To train models, use the `run/train_models.py` script. However, we recommend using the `run/launch_experiment.sh` script instead of directly running the Python file. This ensures that all hyperparameters and configurations are properly managed.

You can control all the hyperparameters for training in the `run/launch_experiment.sh` file. Additionally, you need to explicitly specify whether to run the experiment locally (`run_here`) or submit it to a cluster (`submit`). This will fully train many mixture of experts and then create predictions and plots associated with training and testing.

**NOTE:**
Do not forget to replace `paths` in `run/launch_experiment.sh` and `run/run_training.slurm` with the proper paths to your data and root directory, additionally remember to download the CIFAR-10 dataset to run these experiments.

For example:

```bash
bash run/launch_experiment.sh run_here
```

or

```bash
bash run/launch_experiment.sh submit
```

### Generating Plots

To create plots from experiment results, use the `run/create-plots.py` script. For example:

```bash
python run/create-plots.py
```

This script generates visualizations to analyze the performance of the models. However the models must have been previously created, there is little point to this since integrating this plot creation into the main.

## Additional Information

For more details on datasets and experiments, refer to the `data/README.md` and `experiments/README.md` files.
