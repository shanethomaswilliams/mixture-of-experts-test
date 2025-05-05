#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

# DATASET PARAMETERS
export dataset_name="cifar10"
export data_dir="/cluster/tufts/hugheslab/swilli26/MOE_TESTS/data"
export batch_size=512
export num_classes=10
export val_ratio=0.1

# Training configuration
export use_skip="True"
export l2pen_mag=0.1
export n_epochs=500
export do_early_stopping="True"
export patience=20

# Create base directories once
mkdir -p /cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/logs

# Define all experiment configurations
learning_rates=("0.01")
expert_configs=(
    "4:2"
    "1:1"
)

# Define single good values for all weights
export switch_balance_weight=0.5    # Moderate value for load balancing
export importance_weight=0.1        # Small but meaningful for expert specialization
export topk_balance_weight=0.3      # Moderate for top-k routing
export mutual_info_weight=0.2       # Encourages class specialization
export entropy_weight=0.02          # Small entropy penalty

# Calculate total experiments
total_experiments=${#expert_configs[@]}

for lr in "${learning_rates[@]}"; do
    for config in "${expert_configs[@]}"; do
        IFS=':' read -r experts k <<< "$config"
        
        # Setting
        export num_experts=$experts
        export top_k=$k
        export learning_rate=$lr
        
        if [ "$num_experts" != "1" ]; then
            # MoE configuration with fixed weights
            export run_label="num-experts_${num_experts}_topk_${top_k}_lr_${learning_rate}"
            export output_dir="/cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/${run_label}"
        else
            # ResNet18 configuration - zero weights
            export switch_balance_weight=0.0
            export importance_weight=0.0
            export topk_balance_weight=0.0
            export mutual_info_weight=0.0
            export entropy_weight=0.0
            export run_label="ResNet18_lr_${learning_rate}"
            export output_dir="/cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/${run_label}"
        fi
        
        # These need to be full paths for the summary JSON
        export model_filename="${output_dir}/models/model.pth"
        export preds_filename="${output_dir}/preds/preds.pt"
        export fig_filename="${output_dir}/plots/plot.png"
        export results_file="${output_dir}/results.json"
        
        # Create output directories
        mkdir -p ${output_dir}/models
        mkdir -p ${output_dir}/preds
        mkdir -p ${output_dir}/plots

        if [[ $ACTION_NAME == 'list' ]]; then
            echo "-------------------------------------------"
            echo "Experiment: $run_label"
            echo "  Number of experts: $num_experts"
            echo "  Top-k: $top_k"
            echo "  Learning rate: $learning_rate"
            if [ "$num_experts" != "1" ]; then
                echo "  Switch balance weight: $switch_balance_weight"
                echo "  Importance weight: $importance_weight"
                echo "  Top-k balance weight: $topk_balance_weight"
                echo "  Mutual info weight: $mutual_info_weight"
                echo "  Entropy weight: $entropy_weight"
            fi
            echo "-------------------------------------------"
        elif [[ $ACTION_NAME == 'submit' ]]; then
            sbatch run_training.slurm
        elif [[ $ACTION_NAME == 'run_here' ]]; then
            bash run_training.slurm
        fi
    done
done

if [[ $ACTION_NAME == 'list' ]]; then
    echo "Total experiments: $total_experiments"
elif [[ $ACTION_NAME == 'submit' ]]; then
    echo "All $total_experiments experiments submitted!"
fi