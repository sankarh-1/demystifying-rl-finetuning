# Demystifying Reinforcement Learning Post-Training of Language Models

This repository contains the official code for the paper: **Demystifying Reinforcement Learning Post-Training of Language Models**.

* **Authors:** Donovan Clay, Saket Gollapudi, Sankar Harilal, Min Jang, Jacob Morrison, Sewoong Oh, Natasha Jaques.
* **Affiliations:** University of Washington, Allen Institute for AI.
* **Website:** https://minjang10.github.io/demystifying-rl-finetuning-web/
* **Code:** https://github.com/sankarh-1/demystifying-rl-finetuning

---

## Overview

Reinforcement learning (RL) post-training has emerged as a powerful framework for enhancing the capabilities of large language models (LLMs), yet the principles behind classical RL often remain a "black box" for many practitioners. This repository provides the codebase to deconstruct the RL post-training algorithm by isolating its mechanics in a controlled and simplified environment.

By separating the pipeline into controllable experimental sandboxes, we investigate how RL outcomes are fundamentally shaped by:

* **Base Model Distribution:** How the base model's prior distribution and initial probability mass on a desired behavior govern post-training success.
* **Reward Granularity:** How the success of learning new behaviors changes when shifting from sparse optimal rewards to dense proxy rewards.
* **Prompt Distribution:** How the diversity of the prompt distribution interacts with spurious or random rewards to affect model entropy and capabilities.

The repository is divided into two main sub-folders to test these dynamics on sequence generation and mathematical reasoning.

---

## 1. RLHF Target Quote Memorization Pipeline (`/movie-quotes`)

This sub-folder contains an end-to-end experimental pipeline designed to test the behavior of LLMs under different Fine-Tuning and RLHF strategies. Specifically, the pipeline investigates how a model learns to generate a specific `TARGET_QUOTE` by comparing standard Supervised Fine-Tuning (SFT) against PPO-based Reinforcement Learning, using both Sparse and Levenshtein reward functions.

### Setup

It tests RL performance across 6 distinct combinations:

* **Source Models:** `base`, `sft_plus` (positively biased), `sft_minus` (negatively biased).
* **Reward Functions:** `sparse` (`rl_sparse.py`), `levenshtein` (`rl_lev.py`).

### Codebase Architecture & Execution Flow

The entire experiment is orchestrated by `run_exp.py`, which manages subprocesses, checks for existing checkpoints to allow resuming, and handles the FastAPI server lifecycle.

1. **Data Preparation (`prepare_data.py`):** Generates an 80/20 mixed SFT dataset (distractors vs. target quote) and empty prompts for RL rollouts.
2. **Base Evaluation (`analytics.py`):** Establishes the zero-shot baseline of the model. `baseline.py` is also available as a standalone script for intermediate pure base testing.
3. **SFT+ Training (`sft_plus.py`):** Standard LoRA fine-tuning on the target quote using DeepSpeed and OpenRLHF.
4. **SFT- Training (`sft_minus.py`):** Implements a custom `NegativeTrainer` that flips the Cross-Entropy loss to actively teach the model to avoid generating the target quote.
5. **Reward Server Initialization (`reward_server.py`):** Boots up a local FastAPI server on port 5000 to serve live RL rewards, computing rewards using `rl_utils.py` and streaming live moving averages to W&B.
6. **The RL Matrix (`rl_sparse.py`, `rl_lev.py`):** Utilizes `openrlhf.cli.train_ppo_ray` to distribute Actor, Critic, and Reference models. PPO is trained against a sparse reward signal (0.0 for no match, 1.0 for exact match) and a dense reward signal (normalized Levenshtein distance).
7. **Analytics (`analytics.py`):** Evaluates every checkpoint for exact matches, substring matches, and generation entropy (to measure model collapse/confidence), logging rich tables to Weights & Biases (W&B).

### Usage

1. **Configure your environment:** Ensure your `config.py` is properly set up with your desired `MODEL_NAME`, `TARGET_QUOTE`, and hyperparameters.

2. **Login to W&B:** The pipeline relies heavily on Weights & Biases for live logging. Run:

   ```bash
   wandb login
   ```

3. **Run the Pipeline:**

   ```bash
   python run_exp.py
   ```

### Outputs and Artifacts

As the pipeline runs, it generates several artifacts in the directories defined by your `config.py` (typically `results/`):

* **Models:** Saved LoRA adapters and merged models in `results/models/`.
* **W&B Dashboards:** Live tracking of PPO rewards, exact match percentages, loss curves, and generation tables.
* **Logs:** Raw and cleaned text generations are appended to `results/stats/all_generations.txt`.
* **Summary Report:** Once the 3x2 matrix is complete, a final `summary_report.txt` is generated detailing the Exact Match %, Substring Match %, and Entropy across all 9 evaluation checkpoints.

---

## 2. Math Reasoning RLVR (`/math-reasoning`)

This sub-folder scales the controlled setup to combinatorial mathematical reasoning. It runs a base -> SFT+ / SFT- -> RL (sparse and dense) matrix on a single AIME problem, scoring rollouts with a locally hosted LLM judge.

### Running

```bash
bash run.sh
```

`run.sh` invokes `run_exp.py`, which orchestrates every phase and skips any that already have a checkpoint under `results/models/` or an entry in `results/stats/stats_history.json`.

Model, prompts, and hyperparameters live in `config.py`.

### Environment Variables

All are optional and have working defaults.

Note: The judge must be reachable at `http://$JUDGE_HOST:$JUDGE_PORT/v1` before the RL phases start. `run.sh` adds `$JUDGE_HOST` to `NO_PROXY` so proxied clusters can still reach it directly.

| **Variable**              | **Default**                | **Purpose**                                                      |
| ------------------------- | -------------------------- | ---------------------------------------------------------------- |
| `JUDGE_HOST`              | `127.0.0.1`                | Host serving the OpenAI-compatible judge model                   |
| `JUDGE_PORT`              | `8001`                     | Port for the judge server                                        |
| `REWARD_PORT`             | `5000`                     | Port the local reward server binds to                            |
| `OPENRLHF_DEEPSPEED_PATH` | active env's site-packages | OpenRLHF `deepspeed.py` patched at startup                       |
| `CLUSTER_PROXY`           | unset                      | HTTP(S) proxy; the proxy block in `run.sh` is skipped when unset |
| `WANDB_API_KEY`           | unset                      | Set in `run.sh` before wandb login                               |

## Key Findings Demonstrated in this Codebase

Running the pipelines in this repository will reproduce the core findings of our paper:

* **The Coverage Principle:** The success of RL post-training depends heavily on whether the base model already places sufficient probability mass on the desired behavior. Behaviors with negligible initial probability are difficult to discover and reinforce under sparse rewards.

* **Dense Reward Shaping:** By using a sufficiently dense and correct reward function (such as Levenshtein distance for string matching or Process Reward Models for multi-step math), RL post-training can successfully teach a model new behaviors that have near-zero support in the original base model's distribution.

* **Prompt Distribution Dictates Spurious Reward Effects:** Post-training with spurious or random rewards on a narrow prompt distribution can improve performance in that specific domain if the base model is already strongly biased toward it. However, training with random rewards on a broad prompt distribution uniformly increases policy entropy and degrades overall capabilities.

## Prompt Distribution Runs
To run the spurious rewards runs for the prompt distribution experiments, run the following commands
```bash
cd Spurious_Rewards

# Create the conda environment
conda create -n spurious-rewards python=3.10
pip install -r requirements.txt

# [OPTIONAL]: set wandb credentials
export WANDB_API_KEY=...

# run the spurious rewards from narrow prompt distribution experiments
bash run_narrow.sh

# run the spurious rewards from broad prompt distribution experiments
bash run_broad.sh
```
On WandB, the entropy of the models are tracked, which may be useful in interpreting the results.

## Citation

If you utilize this codebase or our findings, please cite our work:

```bibtex
@misc{clay2026demystifying,
  title={Demystifying Reinforcement Learning Post-Training of Language Models},
  author={Donovan Clay and Saket Gollapudi and Sankar Harilal and Min Jang and Jacob Morrison and Sewoong Oh and Natasha Jaques},
  year={2026},
  url={https://minjang10.github.io/demystifying-rl-finetuning-web/}
}
```
