# Math Reasoning RLVR

Runs a base → SFT+ / SFT− → RL (sparse and dense) matrix on a single AIME problem,
scoring rollouts with a locally hosted LLM judge.

## Running

    bash run.sh

`run.sh` invokes `run_exp.py`, which orchestrates every phase and skips any that
already have a checkpoint under `results/models/` or an entry in
`results/stats/stats_history.json`.

Model, prompts, and hyperparameters live in `config.py`.

## Environment variables

All are optional and have working defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `JUDGE_HOST` | `127.0.0.1` | Host serving the OpenAI-compatible judge model |
| `JUDGE_PORT` | `8001` | Port for the judge server |
| `REWARD_PORT` | `5000` | Port the local reward server binds to |
| `OPENRLHF_DEEPSPEED_PATH` | active env's site-packages | OpenRLHF `deepspeed.py` patched at startup |
| `CLUSTER_PROXY` | unset | HTTP(S) proxy; the proxy block in `run.sh` is skipped when unset |
| `WANDB_API_KEY` | unset | Set in `run.sh` before `wandb login` |

The judge must be reachable at `http://$JUDGE_HOST:$JUDGE_PORT/v1` before the RL
phases start. `run.sh` adds `$JUDGE_HOST` to `NO_PROXY` so proxied clusters can
still reach it directly.
