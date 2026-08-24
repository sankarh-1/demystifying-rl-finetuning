from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
import config
import rl_utils
import wandb
import collections
import os
import json
import re
import ast
import asyncio
from openai import AsyncOpenAI

app = FastAPI()
active_run = None
log_lock = asyncio.Lock()

aclient = AsyncOpenAI(api_key="EMPTY", base_url=config.JUDGE_URL, timeout=120.0)

class RewardRequest(BaseModel):
    query: List[str]
    prompts: List[str]
    labels: Optional[List[Any]] = None

step_counters = {"sparse": 0, "lev": 0}
rolling_metrics = {
    "sparse_match": collections.deque(maxlen=40), "sparse_reward": collections.deque(maxlen=40),
    "lev_match": collections.deque(maxlen=40), "lev_reward": collections.deque(maxlen=40)
}

def switch_wandb_run(phase_type):
    global active_run
    base_run_name = os.environ.get("LIVE_WANDB_NAME", f"unknown_{phase_type}_live")
    full_run_name = f"{base_run_name}_live"
    if active_run != full_run_name:
        if wandb.run is not None: wandb.finish()
        wandb.init(project=config.WANDB_PROJECT_NAME, name=full_run_name, reinit=True, settings=wandb.Settings(init_timeout=300))
        active_run = full_run_name
        step_counters[phase_type] = 0
        rolling_metrics[f"{phase_type}_match"].clear()
        rolling_metrics[f"{phase_type}_reward"].clear()

def extract_json_from_text(text: str) -> dict:
    md_match = re.search(r'`{3}(?:json)?\s*(\{.*?\})\s*`{3}', text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except Exception:
            pass
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        chunk = text[start_idx:end_idx + 1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(chunk)
            except (ValueError, SyntaxError):
                pass
                
    raise ValueError("No valid JSON dictionary brackets found in the response.")

async def evaluate_multi_path_reasoning_async(trace: str):
    system_prompt = (
        "You are an expert mathematical proof grader. Evaluate the student's solution to:\n"
        "12x^2 - xy - 6y^2 = 0 where x, y are integers bounded between -100 and 100 inclusive.\n\n"
        "Step 1: Classify their strategy strictly into one of these paths:\n"
        "- 'Path 1': Direct Factoring into (3x + 2y)(4x - 3y) = 0.\n"
        "- 'Path 2': De-homogenization (dividing by y^2).\n"
        "- 'Path 3': Quadratic Formula.\n"
        "- 'Other': Any other valid or divergent logical attempt (including writing Python code).\n\n"
        "Step 2: Evaluate the following milestones and mark them true or false:\n"
        "m1: Initiated a correct algebraic path configuration.\n"
        "m2: Derived both accurate base root relationships.\n"
        "m3: Evaluated boundary constraints correctly.\n"
        "m4: Calculated correct solution counts for at least one case branch.\n"
        "m5: Addressed the intersection overlap at the origin (0,0) correctly.\n\n"
        "CRITICAL INSTRUCTION: Return ONLY a raw JSON object formatted exactly as below. Do NOT add any extra text, markdown, or explanations.\n"
        '{"path": "Path 1", "m1": true, "m2": true, "m3": false, "m4": false, "m5": false}'
    )
    
    raw_response = ""
    extracted = rl_utils.extract_boxed_answer(trace)
    
    try:
        response = await aclient.chat.completions.create(
            model=config.JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Student Work Trace:\n{trace}"}
            ],
            temperature=0.0,
            max_tokens=512
        )
        
        raw_response = response.choices[0].message.content
        scores = extract_json_from_text(raw_response)
        
        process_bonus = 0.0
        if scores.get("m1", False): process_bonus += 0.05
        if scores.get("m2", False): process_bonus += 0.05
        if scores.get("m3", False): process_bonus += 0.10
        if scores.get("m4", False): process_bonus += 0.15
        if scores.get("m5", False): process_bonus += 0.25
        
        if extracted == config.TARGET_ANSWER:
            total_reward = 1.0
        elif extracted == "118":
            total_reward = 0.6
        else:
            total_reward = process_bonus
        
        if "\\boxed{" not in trace:
            total_reward -= 0.5
            
        total_reward = max(0.0, float(total_reward))
            
        return total_reward, scores.get("path", "Other"), extracted
        
    except Exception as e:
        error_name = type(e).__name__
        print(f"\n{'!' * 60}")
        print(f"JUDGE API / PARSE ERROR: {error_name}")
        print(f"MSG: {e}")
        print(f"RAW JUDGE OUTPUT THAT CAUSED CRASH:\n{raw_response}")
        print(f"{'!' * 60}")
        
        async with log_lock:
            with open(config.STATS_DIR / "judge_raw_responses.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'!' * 80}\n")
                f.write(f"CRASH: {error_name} - {e}\n")
                f.write(f"{'-' * 40} TRACE THAT CAUSED CRASH {'-' * 40}\n")
                f.write(trace + "\n")
                f.write(f"{'-' * 40} JUDGE RAW OUTPUT {'-' * 40}\n")
                f.write(raw_response + "\n")
                f.write(f"{'!' * 80}\n")
        
        return 0.0, f"ERROR: {error_name}", extracted

@app.post("/get_reward_sparse")
async def get_reward_sparse(req: RewardRequest):
    switch_wandb_run("sparse")
    
    traces = []
    for q, p in zip(req.query, req.prompts):
        clean_trace = q[len(p):].split("<|endoftext|>")[0].split("<|im_end|>")[0].strip()
        traces.append(clean_trace)
        
    eval_results = await asyncio.gather(*[evaluate_multi_path_reasoning_async(t) for t in traces])
    
    rewards, live_table_data = [], []
    for trace, (val, path, extracted) in zip(traces, eval_results):
        sparse_val = rl_utils.compute_sparse_math_reward(trace, config.TARGET_ANSWER)
        is_match = 1 if sparse_val == 1.0 else 0
        total_sparse = 1.0 if is_match == 1 else 0.0
        
        if "\\boxed{" not in trace:
            total_sparse -= 0.5
            
        total_sparse = max(0.0, float(total_sparse))
            
        rewards.append(total_sparse)
        rolling_metrics["sparse_match"].append(is_match)
        rolling_metrics["sparse_reward"].append(total_sparse)
        
        live_table_data.append([trace, path, extracted, is_match, total_sparse])
        
    if step_counters["sparse"] >= 0:
        rolling_pct = (sum(rolling_metrics["sparse_match"]) / len(rolling_metrics["sparse_match"])) * 100
        rolling_rew = sum(rolling_metrics["sparse_reward"]) / len(rolling_metrics["sparse_reward"])
        
        live_table = wandb.Table(columns=["Full Sequence Text", "Strategy Path Tagged", "Extracted Answer", "Verifiable Correct", "Sparse Score"], data=live_table_data)
        
        try: 
            wandb.log({"step": step_counters["sparse"], "exact_match_pct": rolling_pct, "avg_reward": rolling_rew, "live_generations": live_table})
        except Exception as e: 
            print(f"W&B Logging Skipped: {e}")
    
    step_counters["sparse"] += 1
    return {"rewards": rewards}

@app.post("/get_reward_lev")
async def get_reward_lev(req: RewardRequest):
    switch_wandb_run("lev")
    
    traces = []
    for q, p in zip(req.query, req.prompts):
        clean_trace = q[len(p):].split("<|endoftext|>")[0].split("<|im_end|>")[0].strip()
        traces.append(clean_trace)
        
    print(f"\nDISPATCHING {len(traces)} TRACES TO 32B JUDGE...")
    eval_results = await asyncio.gather(*[evaluate_multi_path_reasoning_async(t) for t in traces])
    
    rewards, live_table_data = [], []
    for trace, (val, path, extracted) in zip(traces, eval_results):
        rewards.append(val)
        is_match = 1 if extracted == config.TARGET_ANSWER else 0
        rolling_metrics["lev_match"].append(is_match)
        rolling_metrics["lev_reward"].append(val)
        
        live_table_data.append([trace, path, extracted, is_match, val])
        
    if step_counters["lev"] >= 0:
        rolling_pct = (sum(rolling_metrics["lev_match"]) / len(rolling_metrics["lev_match"])) * 100
        rolling_rew = sum(rolling_metrics["lev_reward"]) / len(rolling_metrics["lev_reward"])
        
        live_table = wandb.Table(columns=["Full Sequence Text", "Strategy Path Tagged", "Extracted Answer", "Exact Box Match", "Dense Reward"], data=live_table_data)
        
        try: 
            wandb.log({"step": step_counters["lev"], "exact_match_pct": rolling_pct, "avg_reward": rolling_rew, "live_generations": live_table})
        except Exception as e: 
            print(f"W&B Logging Skipped: {e}")
            
    step_counters["lev"] += 1
    return {"rewards": rewards}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.REWARD_PORT)
