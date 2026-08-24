#!/bin/bash

if command -v module >/dev/null 2>&1; then
    module load cuda/12.4
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    echo "Environment loaded. CUDA_HOME: $CUDA_HOME"
fi

export WANDB_API_KEY="ENTER WANDB KEY HERE"
wandb login

python3 run_exp.py
