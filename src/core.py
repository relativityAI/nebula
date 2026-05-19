import os
import json
import time
import re
import random
import string
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime
from db.models import Profile, AnalysisRun
import pandas as pd

# Available data source structures
DATA_SOURCE_TEMPLATES = {
    "screener": {
        "pe": 0.0,
        "market_cap": 0.0,
        "dividend_yield": 0.0,
        "debt_to_equity": 0.0
    },
    "trendlyne": {
        "momentum_score": 0,
        "durability_score": 0,
        "valuation_score": 0
    },
    "yfinance": {
        "beta": 0.0,
        "fifty_two_week_high": 0.0,
        "fifty_two_week_low": 0.0
    }
}

async def create_template_logic(name: str, sources: List[str]):
    template = {
        "name": name,
        "data_sources": {}
    }
    for source in sources:
        template["data_sources"][source] = DATA_SOURCE_TEMPLATES.get(source, {"custom_field": None})
    
    os.makedirs("templates", exist_ok=True)
    filename = f"templates/{name}.json"
    with open(filename, "w") as f:
        json.dump(template, f, indent=4)
    return filename

async def save_profile_logic(data: Dict[str, Any]):
    name = data.get("name")
    if not name:
        raise ValueError("Profile name is missing")

    profile = await Profile.find_one({"name": name})
    if profile:
        profile.data_sources = data.get("data_sources", {})
        profile.parameters = data.get("parameters", {})
        await profile.save()
    else:
        profile = Profile(
            name=name,
            data_sources=data.get("data_sources", {}),
            parameters=data.get("parameters", {})
        )
        await profile.insert()
    return profile

async def get_available_profiles_logic():
    return await Profile.find_all().to_list()

async def get_profile_logic(name: str):
    return await Profile.find_one({"name": name})

async def get_analysis_runs_logic():
    return await AnalysisRun.find_all().sort(+AnalysisRun.created_at).to_list()

async def run_correlation_logic(
    share_name: str,
    symbol: str,
    profile_name: str,
    model: str = "cerebras/qwen-3-32b",
    corr_id: str = None,
    iters: int = 1,
    rpm: int = 2,
    max_retry: int = 3
):
    from src.technicals import get_price_data, technical_analysis_talib
    from src.agents import NebulAgent

    def short_id(n=2):
        ts = datetime.utcnow().strftime("%y%m%d%H%M%S")
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
        return f"{ts}{rand}"

    def extract_score(text):
        m = re.search(r'FINAL_SCORE:\s*(\d{1,3})', text)
        return int(m.group(1)) if m else None

    start_time = time.time()
    corr_id = corr_id or short_id()
    
    profile = await Profile.find_one({"name": profile_name})
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found")

    run = await AnalysisRun.find_one({"corr_id": corr_id})
    if run:
        run.rpm = rpm
        run.max_retry = max_retry
        run.iterations = max(run.iterations, iters)
    else:
        run = AnalysisRun(
            symbol=symbol,
            share_name=share_name,
            profile=profile_name,
            model=model,
            corr_id=corr_id,
            iterations=iters,
            rpm=rpm,
            max_retry=max_retry
        )

    config_data = profile.data_sources if not profile.parameters else profile.parameters

    price_df = get_price_data(f"{symbol}.NS", "3y")
    run.technical_analysis = technical_analysis_talib(symbol, price_df)
    await run.save()

    q = []
    agent = NebulAgent(symbol=symbol, profile=profile_name, model=model)

    for i in range(iters):
        iteration = str(i + 1)
        if iteration not in run.runs:
            run.runs[iteration] = {"iteration": iteration, "start_time": time.time(), "parameters": {}}
        
        for param, config in config_data.items():
            if param in run.runs[iteration]['parameters']:
                continue

            # Rate limiting
            now = time.time()
            q = [t for t in q if t > now - 60]
            if len(q) >= rpm:
                time.sleep(60 - (now - q[0]))
            q.append(time.time())

            title = config.get('title', param) if isinstance(config, dict) else param
            desc = config.get('description', str(config)) if isinstance(config, dict) else str(config)
            tools = config.get('tools', ["read_latest_transcript"]) if isinstance(config, dict) else ["read_latest_transcript"]
            
            response = agent.run(title, desc, tools)
            score = extract_score(response.choices[0].message.content)
            
            run.runs[iteration]['parameters'][param] = {
                "score": score,
                "weight": float(config.get('weight', 1.0)) if isinstance(config, dict) else 1.0,
                "response": response.model_dump()
            }
            run.duration = time.time() - start_time
            run.end_time = time.time()
            await run.save()
            
    return run

async def get_run_scores_logic(corr_id: str):
    run = await AnalysisRun.find_one({"corr_id": corr_id})
    if not run:
        return None
    
    profile = await Profile.find_one({"name": run.profile})
    
    dfs = []
    for iter_id, iteration in run.runs.items():
        params, scores = [], []
        for p, data in iteration['parameters'].items():
            params.append(p)
            scores.append(data['score'])
        dfs.append(pd.DataFrame({'parameter': params, iter_id: scores}).set_index('parameter'))

    df = pd.concat(dfs, axis=1)
    df['avg'] = df.mean(axis=1).astype(int)
    
    qual_config = profile.parameters.get('qualitative', {}) or profile.data_sources
    weights = {p: qual_config[p].get('weight', 1.0) if isinstance(qual_config[p], dict) else 1.0 for p in qual_config}
    avgs = df['avg']
    
    total_weight = sum(weights[p] for p in avgs.index if p in weights)
    qual_score = sum(weights[p] * avgs[p] for p in avgs.index if p in weights) / total_weight if total_weight > 0 else 0
    
    return {
        "run": run,
        "dataframe": df,
        "qualitative_score": round(qual_score, 2)
    }

async def delete_profile_logic(name: str):
    profile = await Profile.find_one({"name": name})
    if profile:
        await profile.delete()
        return True
    return False
