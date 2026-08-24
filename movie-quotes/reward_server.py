from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
import config
import rl_utils
import wandb
import collections
import os

app = FastAPI()
active_run = None

class RewardRequest(BaseModel):
    query: List[str]
    prompts: List[str]
    labels: Optional[List[Any]] = None

step_counters = {"sparse": 0, "lev": 0}

rolling_metrics = {
    "sparse_match": collections.deque(maxlen=40),
    "sparse_reward": collections.deque(maxlen=40),
    "lev_match": collections.deque(maxlen=40),
    "lev_reward": collections.deque(maxlen=40)
}

def switch_wandb_run(phase_type):
    global active_run
    base_run_name = os.environ.get("LIVE_WANDB_NAME", f"unknown_{phase_type}_live")
    model_str = config.MODEL_NAME.split('/')[-1]
    full_run_name = f"{base_run_name}_{model_str}_live"
    
    if active_run != full_run_name:
        if wandb.run is not None: wandb.finish()
        wandb.init(project=config.WANDB_PROJECT_NAME, name=full_run_name, reinit=True)
        active_run = full_run_name
        step_counters[phase_type] = 0
        rolling_metrics[f"{phase_type}_match"].clear()
        rolling_metrics[f"{phase_type}_reward"].clear()

@app.post("/get_reward_sparse")
def get_reward_sparse(req: RewardRequest):
    switch_wandb_run("sparse")
    rewards = []
    live_table_data = []
    target_clean = rl_utils.normalize_text(config.TARGET_QUOTE)
    
    for q, p in zip(req.query, req.prompts):
        response = q[len(p):] 
        clean_text = rl_utils.extract_quote_content(response)
        
        is_match = 1 if target_clean == clean_text else 0
        val = rl_utils.compute_sparse_reward(clean_text, target_clean)
        
        rolling_metrics["sparse_match"].append(is_match)
        rolling_metrics["sparse_reward"].append(val)
        rewards.append(val)
        
        live_table_data.append([clean_text, is_match, val])
        
    if step_counters["sparse"] > 0 and step_counters["sparse"] % 10 == 0:
        rolling_pct = (sum(rolling_metrics["sparse_match"]) / len(rolling_metrics["sparse_match"])) * 100
        rolling_rew = sum(rolling_metrics["sparse_reward"]) / len(rolling_metrics["sparse_reward"])
        live_table = wandb.Table(columns=["Cleaned Output", "Exact Match", "Sparse Reward"], data=live_table_data)
        
        wandb.log({
            "step": step_counters["sparse"], 
            "exact_match_pct": rolling_pct, 
            "avg_reward": rolling_rew,
            "live_generations": live_table
        })
        
    step_counters["sparse"] += 1
    return {"rewards": rewards}

@app.post("/get_reward_lev")
def get_reward_lev(req: RewardRequest):
    switch_wandb_run("lev")
    rewards = []
    live_table_data = []
    target_clean = rl_utils.normalize_text(config.TARGET_QUOTE)
    
    for q, p in zip(req.query, req.prompts):
        response = q[len(p):] 
        clean_text = rl_utils.extract_quote_content(response)
        
        is_match = 1 if target_clean == clean_text else 0
        val = rl_utils.compute_levenshtein_reward(clean_text, target_clean)
        
        rolling_metrics["lev_match"].append(is_match)
        rolling_metrics["lev_reward"].append(val)
        rewards.append(val)
        
        live_table_data.append([clean_text, is_match, val])
        
    if step_counters["lev"] > 0 and step_counters["lev"] % 10 == 0:
        rolling_pct = (sum(rolling_metrics["lev_match"]) / len(rolling_metrics["lev_match"])) * 100
        rolling_rew = sum(rolling_metrics["lev_reward"]) / len(rolling_metrics["lev_reward"])
        live_table = wandb.Table(columns=["Cleaned Output", "Exact Match", "Lev Reward"], data=live_table_data)
        
        wandb.log({
            "step": step_counters["lev"], 
            "exact_match_pct": rolling_pct, 
            "avg_reward": rolling_rew,
            "live_generations": live_table
        })
        
    step_counters["lev"] += 1
    return {"rewards": rewards}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.REWARD_PORT)
