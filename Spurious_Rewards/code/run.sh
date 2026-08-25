conda activate spurious-rewards

# A40s are 44GB; the actor shares GPU 0 with the colocated ref shard, so the
# allocator runs hot. Reduces fragmentation-driven OOMs in the ZeRO-3 backward.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder; CPUAdamBuilder().load()"

# For the narrow prompt distribution run
bash scripts/rlvr_deepscaler_grpo_qwen_random.sh

# For the broad prompt distribution run
bash scripts/rlvr_deepscaler_grpo_qwen_random_broad.sh