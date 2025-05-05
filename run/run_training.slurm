#!/usr/bin/env bash
#SBATCH -p ccgpu
#SBATCH --gres=gpu
#SBATCH --job-name=MoE_Test
#SBATCH --output=/cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/logs/moe_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/logs/moe_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

module load cuda/11.0
source activate Echo_MIL

JOB_ID=${SLURM_JOB_ID}
LOG_DIR="/cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/logs"
OUTPUT_FILE="${LOG_DIR}/moe_${JOB_ID}.out"
ERROR_FILE="${LOG_DIR}/moe_${JOB_ID}.err"

# Pass along all ENV variables as arguments to my Python script
export PYTHONPATH="${PYTHONPATH}:/cluster/tufts/hugheslab/swilli26/MOE_TESTS"
python /cluster/tufts/hugheslab/swilli26/MOE_TESTS/run/train_models.py \
    --dataset_name ${dataset_name:-"cifar10"} \
    --data_dir ${data_dir:-"/cluster/tufts/hugheslab/swilli26/MOE_TESTS/data"} \
    --batch_size ${batch_size:-128} \
    --num_classes ${num_classes:-10} \
    --val_ratio ${val_ratio:-0.1} \
    --num_experts ${num_experts:-4} \
    --use_skip ${use_skip:-True} \
    --top_k ${top_k:-1} \
    --learning_rate ${learning_rate:-0.01} \
    --l2pen_mag ${l2pen_mag:-0.0} \
    --switch_balance_weight ${switch_balance_weight:-0.05} \
    --importance_weight ${importance_weight:-0.05} \
    --topk_balance_weight ${topk_balance_weight:-0.05} \
    --mutual_info_weight ${mutual_info_weight:-0.05} \
    --entropy_weight ${entropy_weight:-0.01} \
    --n_epochs ${n_epochs:-1000} \
    --do_early_stopping ${use_skip:-True} \
    --patience ${patience:-15} \
    --run_label ${run_label:-"num-experts_4_topk_1_lr_0.01"} \
    --output_dir ${output_dir:-"/cluster/tufts/hugheslab/swilli26/MOE_TESTS/experiments/$run_label"} \
    --model_filename ${model_filename:-"model.pth"} \
    --preds_filename ${preds_filename:-"preds.pt"} \
    --fig_filename ${fig_filename:-"plot.png"} \
    --results_file ${results_file:-"results.json"}


# Store the exit code of the Python script
PYTHON_EXIT_CODE=$?

# Copy log files to output directory
if [ -d "${output_dir}" ]; then
    mkdir -p "${output_dir}/logs"
    cp "${OUTPUT_FILE}" "${output_dir}/logs/logs.out"
    cp "${ERROR_FILE}" "${output_dir}/logs/logs.err"
    
    # Delete original files
    rm -f "${OUTPUT_FILE}"
    rm -f "${ERROR_FILE}"
fi

# Exit with the same code as the Python script
exit $PYTHON_EXIT_CODE