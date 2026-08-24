# RLHF Target Quote Memorization Pipeline

This repository contains an end-to-end experimental pipeline designed to test the behavior of Large Language Models (LLMs) under different Fine-Tuning and Reinforcement Learning from Human Feedback (RLHF) strategies. 

Specifically, the pipeline investigates how a model learns to generate a specific `TARGET_QUOTE` by comparing standard Supervised Fine-Tuning (SFT) against PPO-based Reinforcement Learning, using both Sparse and Levenshtein reward functions. It also tests RL performance when starting from a baseline model, a positively biased model (SFT+), and a negatively biased model (SFT-).

## Execution Flow

The entire experiment is orchestrated by **`run_exp.py`**. When executed, the pipeline follows this matrix:

1. **Data Preparation (`prepare_data.py`)**: Generates an 80/20 mixed SFT dataset (distractors vs. target quote) and empty prompts for RL rollouts.
2. **Base Evaluation (`analytics.py`)**: Establishes the zero-shot baseline of the model.
3. **SFT+ Training (`sft_plus.py`)**: Standard LoRA fine-tuning on the target quote.
4. **SFT- Training (`sft_minus.py`)**: Custom un-learning/negative fine-tuning to heavily penalize the generation of the target quote.
5. **Reward Server Initialization (`reward_server.py`)**: Boots up a local FastAPI server on port 5000 to serve live RL rewards.
6. **The 3x2 RL Matrix**: Runs PPO using Ray across 6 distinct combinations:
   * **Source Models**: `base`, `sft_plus`, `sft_minus`
   * **Reward Functions**: `sparse` (`rl_sparse.py`), `levenshtein` (`rl_lev.py`)
7. **Analytics (`analytics.py`)**: Evaluates every checkpoint for exact matches, substring matches, and generation entropy, logging all data to Weights & Biases (W&B).

## Codebase Architecture

### Core Orchestration & Setup
* **`run_exp.py`**: The main execution script. Manages subprocesses, checks for existing checkpoints to allow resuming, and handles the FastAPI server lifecycle.
* **`config.py`**: *(User Defined)* Centralized configuration hub containing hyperparameters, model paths, batch sizes, the `TARGET_QUOTE`, and W&B project names.
* **`prepare_data.py`**: Formats datasets for SFT and PPO using the tokenizer's chat template.

### Supervised Fine-Tuning (SFT)
* **`sft_plus.py`**: Uses DeepSpeed and OpenRLHF to perform standard causal language modeling on the 80/20 dataset.
* **`sft_minus.py`**: Implements a custom `NegativeTrainer` that flips the Cross-Entropy loss (`ce_loss * -config.NEG_CE_SCALE`) to actively teach the model to *avoid* generating the target quote.

### Reinforcement Learning (PPO)
Both RL scripts utilize `openrlhf.cli.train_ppo_ray` to distribute the Actor, Critic, and Reference models.
* **`rl_sparse.py`**: PPO trained against a sparse reward signal (0.0 for no match, 1.0 for exact match, with a slight penalty for extra characters).
* **`rl_lev.py`**: PPO trained against a dense reward signal using normalized Levenshtein distance (0.0 to 1.0 based on character-level similarity).

### Reward & Evaluation Infrastructure
* **`reward_server.py`**: A FastAPI server that the OpenRLHF Ray workers query during generation rollouts. It computes the rewards using `rl_utils.py` and streams live moving averages to W&B.
* **`rl_utils.py`**: Helper functions containing the text normalization logic, regex extraction for `quote: [...]`, and the mathematical implementations of the Sparse and Levenshtein reward functions.
* **`analytics.py`**: The evaluation engine. Generates sequences in batches, calculates response entropy (to measure model collapse/confidence), checks for exact/substring matches, and logs rich tables to W&B.
* **`baseline.py`**: A standalone script used for intermediate or pure base anchor testing without triggering the full orchestration loop.

## Usage

**1. Configure your environment**
Ensure your `config.py` is properly set up with your desired `MODEL_NAME`, `TARGET_QUOTE`, and hyperparameters.

**2. Login to W&B**
The pipeline relies heavily on Weights & Biases for live logging.
```bash
wandb login
```

**3. Run the Pipeline**
```bash
python run_exp.py
```

## Outputs and Artifacts

As the pipeline runs, it generates several artifacts in the directories defined by your `config.py` (typically `results/`):

* **Models**: Saved LoRA adapters and merged models in `results/models/`.
* **W&B Dashboards**: Live tracking of PPO rewards, exact match percentages, loss curves, and generation tables.
* **Logs**: Raw and cleaned text generations are appended to `results/stats/all_generations.txt`.
* **Summary Report**: Once the 3x2 matrix is complete, a final `summary_report.txt` is generated detailing the Exact Match %, Substring Match %, and Entropy across all 9 evaluation checkpoints.