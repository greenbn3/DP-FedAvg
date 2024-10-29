#!/bin/bash

# ===========================
# Initial Configuration
# ===========================

# Exit immediately if a command exits with a non-zero status
set -e

# Optional: Uncomment the following line to print each command before executing it (useful for debugging)
set -x

# ===========================
# Initialize Conda
# ===========================

# Source the conda.sh script to enable conda commands
# This path is based on your conda info output
source "/home/greenbn3/miniconda3/etc/profile.d/conda.sh"

# Activate the desired Conda environment
conda activate dp_fedavg_env

# Verify that the correct environment is activated
echo "Activated Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Environment Path: /home/greenbn3/miniconda3/envs/dp_fedavg_env"
echo ""

# ===========================
# Configuration Parameters
# ===========================

# Define the epsilon values to iterate over
epsilons=("0.01" "0.1" "0.5" "1.0" "10.0" "25.0" "50.0" "none")

# Define the datasets, their corresponding models, and whether to use IID distribution
# Format: "dataset_name:model_type:iid_flag"
datasets_info=(
    "fashion-mnist:cnn:true"
    "femnist:cnn:false"
    "shakespeare:lstm:false"
)

# Define common training parameters
NUM_CLIENTS=2
ROUNDS=100
EPOCHS=1

# Log directory
LOG_DIR="./log"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# ===========================
# Function Definitions
# ===========================

# Function to run training for a specific dataset and epsilon value
run_training() {
    local dataset=$1
    local model=$2
    local iid_flag=$3
    local epsilon=$4

    echo "-------------------------------------------"
    echo "Starting training for Dataset: $dataset | Epsilon: $epsilon"
    echo "-------------------------------------------"

    # Construct the command
    CMD="python3 DP_FL.py --dataset $dataset --num_clients $NUM_CLIENTS --rounds $ROUNDS --epochs $EPOCHS --epsilon $epsilon --model $model"

    # Add the --iid flag if iid_flag is true
    if [ "$iid_flag" == "true" ]; then
        CMD+=" --iid"
    fi

    # Define the log file name
    # Replace 'none' with 'none' string for consistency
    epsilon_str=${epsilon}
    if [ "$epsilon" == "none" ]; then
        epsilon_str="none"
    fi

    log_file="${LOG_DIR}/${dataset}_${epsilon_str}_training.log"

    echo "Executing command:"
    echo "$CMD"
    echo "Logging output to: $log_file"
    echo ""

    # Execute the command and redirect both stdout and stderr to the log file
    $CMD > "$log_file" 2>&1

    echo ""
    echo "Finished training for Dataset: $dataset | Epsilon: $epsilon"
    echo "Log saved to: $log_file"
    echo ""
}

# ===========================
# Main Execution Loop
# ===========================

# Iterate over each dataset
for dataset_entry in "${datasets_info[@]}"; do
    # Split the dataset entry into dataset name, model type, and iid flag
    IFS=':' read -r dataset model iid_flag <<< "$dataset_entry"

    # Iterate over each epsilon value
    for epsilon in "${epsilons[@]}"; do
        run_training "$dataset" "$model" "$iid_flag" "$epsilon"
    done
done

echo "All training processes have been completed successfully!"