#!/bin/bash

if command -v module >/dev/null 2>&1; then
    module load cuda/12.4
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    echo "Environment loaded. CUDA_HOME: $CUDA_HOME"
fi

if [ -n "${CLUSTER_PROXY:-}" ]; then
    export http_proxy="$CLUSTER_PROXY"
    export https_proxy="$CLUSTER_PROXY"
    export HTTP_PROXY="$CLUSTER_PROXY"
    export HTTPS_PROXY="$CLUSTER_PROXY"
fi

export NO_PROXY="${JUDGE_HOST:-127.0.0.1},localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY="ADD WANDB KEY HERE"
wandb login

killall python python3

python3 run_exp.py
