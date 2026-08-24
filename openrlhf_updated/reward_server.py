from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
import config
import rl_utils
import wandb
import collections

app = FastAPI()
wandb.init(project="quote-rlhf", name="live_match_tracking", reinit=True)

class RewardRequest(BaseModel):
    query: List[str]
    prompts: List[str]
    labels: Optional[List[Any]] = None

step_counters = {"sparse": 0, "lev": 0}

rolling_matches = {
    "sparse": collections.deque(maxlen=40),
    "lev": collections.deque(maxlen=40)
}

@app.post("/get_reward_sparse")
def get_reward_sparse(req: RewardRequest):
    rewards = []
    target_clean = rl_utils.normalize_text(config.TARGET_QUOTE)
    
    for q, p in zip(req.query, req.prompts):
        response = q[len(p):] 
        clean_text = rl_utils.extract_quote_content(response)
        
        # Track Exact Match (1 or 0) for the rolling average
        is_match = 1 if target_clean in clean_text else 0
        rolling_matches["sparse"].append(is_match)
        
        # Calculate actual RL Reward
        val = rl_utils.compute_sparse_reward(clean_text, target_clean)
        rewards.append(val)
        
    if step_counters["sparse"] > 0 and step_counters["sparse"] % 10 == 0:
        rolling_pct = (sum(rolling_matches["sparse"]) / len(rolling_matches["sparse"])) * 100
        wandb.log({"step": step_counters["sparse"], "rl_sparse_exact_match_pct": rolling_pct})
        
    step_counters["sparse"] += 1
    return {"rewards": rewards}

@app.post("/get_reward_lev")
def get_reward_lev(req: RewardRequest):
    rewards = []
    target_clean = rl_utils.normalize_text(config.TARGET_QUOTE)
    
    for q, p in zip(req.query, req.prompts):
        response = q[len(p):] 
        clean_text = rl_utils.extract_quote_content(response)
        
        # Track Exact Match (1 or 0) for the rolling average
        is_match = 1 if target_clean in clean_text else 0
        rolling_matches["lev"].append(is_match)
        
        # Calculate actual RL Reward (Levenshtein)
        val = rl_utils.compute_levenshtein_reward(clean_text, target_clean)
        rewards.append(val)
        
    if step_counters["lev"] > 0 and step_counters["lev"] % 10 == 0:
        rolling_pct = (sum(rolling_matches["lev"]) / len(rolling_matches["lev"])) * 100
        wandb.log({"step": step_counters["lev"], "rl_lev_exact_match_pct": rolling_pct})
        
    step_counters["lev"] += 1
    return {"rewards": rewards}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.REWARD_PORT)
