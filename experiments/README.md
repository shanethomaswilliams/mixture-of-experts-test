# Experiments Directory

This folder stores configurations and results for experiments. Below are instructions on how to run experiments and populate this folder.

## Running Experiments

To run experiments, use the `run/launch_experiment.sh` script. Run the following command:

```bash
bash run/launch_experiment.sh
```

This script will execute the training and evaluation pipelines based on predefined configurations.

## Folder Contents

After running experiments, this folder will contain:

- `[NAME-OF-EXPERIMENT-HYPERPARAMETERS]/`: Contains results and logs from experiments.
  - **Model Files**: Saved models from training.
  - **Plots**: Visualizations generated from experiment results.
  - **Predictions**: Output predictions from the models.
  - **Logs**: Detailed logs of the training and evaluation processes.

Ensure that the `experiments/` folder is properly organized to facilitate analysis and reproducibility.