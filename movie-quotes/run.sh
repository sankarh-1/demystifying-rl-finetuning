#!/bin/bash

# Load the environment
module load cuda/12.4
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "Environment loaded. CUDA_HOME: $CUDA_HOME"

# Authenticate Weights & Biases non-interactively
export WANDB_API_KEY="ENTER WANDB KEY HERE"
wandb login

# Run the pipeline
python3 run_exp.py
