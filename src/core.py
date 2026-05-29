import os
import json
import time
import re
import random
import string
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime
from src.db.models import Profile, AnalysisRun
import pandas as pd
import httpx
import asyncio

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

async def fetch_voyager_data(source: str, symbol: str):
    url = f"http://localhost:8001/{source}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params={"symbol": symbol}, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Voyager API error for {source}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error calling Voyager API for {source}: {str(e)}")
    return None

def evaluate_filter(value, filter_cfg):
    try:
        val = float(value)
        direction = filter_cfg.get('direction', 'higher')
        lower = float(filter_cfg.get('lower', 0))
        upper = float(filter_cfg.get('upper', 0))
        
        if direction == 'higher':
            if val >= upper: return 1.0
            if val <= lower: return 0.0
            return (val - lower) / (upper - lower) if upper != lower else 0.0
        elif direction == 'lower':
            if val <= lower: return 1.0
            if val >= upper: return 0.0
            return (upper - val) / (upper - lower) if upper != lower else 0.0
    except (ValueError, TypeError):
        return 0.0
    return 0.0

async def perform_analysis_task(corr_id: str):
    try:
        run = await AnalysisRun.find_one({"corr_id": corr_id})
        if not run:
            logger.error(f"AnalysisRun {corr_id} not found for background task")
            return

        run.status = "RUNNING"
        await run.save()

        profile = await Profile.find_one({"name": run.profile})
        if not profile:
            run.status = "FAILED"
            run.error = "Profile not found"
            await run.save()
            return

        # Quantitative Analysis
        quant_results = {}
        total_weighted_score = 0.0
        total_weight = 0.0

        for source_cfg in profile.data_sources:
            source_name = source_cfg.get("source")
            if not source_name: continue
            
            data = await fetch_voyager_data(source_name, run.symbol)
            if not data:
                quant_results[source_name] = {"error": "No data from source"}
                continue
            
            source_results = {"metrics": {}}
            source_score = 0.0
            source_metrics_count = 0

            for filter_cfg in source_cfg.get("filters", []):
                metric = filter_cfg.get("metric")
                # Handle cases where the data key might be different (e.g. lowercase or slightly modified)
                # For now, assume exact match or look for it
                val = data.get(metric)
                if val is None:
                    # Fallback search if exact match fails
                    for k, v in data.items():
                        if k.lower() == metric.lower():
                            val = v
                            break
                
                score = evaluate_filter(val, filter_cfg)
                source_results["metrics"][metric] = {
                    "value": val,
                    "score": score
                }
                source_score += score
                source_metrics_count += 1
            
            if source_metrics_count > 0:
                source_results["score"] = source_score / source_metrics_count
                # For a simple scoring system, we can assign weights to sources later if needed
                # For now, let's just average them
                total_weighted_score += source_results["score"]
                total_weight += 1
            
            quant_results[source_name] = source_results

        run.quantitative_analysis = quant_results
        run.total_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Qualitative (Agentic) - The user said ignore for now or keep at 0
        # If we wanted to keep the old logic, we could run it here.
        # But per request: "ignore the qualitative thing... qualitative score may be 0"
        # We can still run it if it doesn't take too long, but let's prioritize the quantitative part.
        
        run.status = "COMPLETED"
        run.end_time = time.time()
        run.duration = run.end_time - run.created_at.timestamp()
        await run.save()
        logger.info(f"Analysis {corr_id} completed successfully")

    except Exception as e:
        logger.exception(f"Error in perform_analysis_task: {str(e)}")
        if run:
            run.status = "FAILED"
            run.error = str(e)
            await run.save()

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
    def short_id(n=2):
        ts = datetime.utcnow().strftime("%y%m%d%H%M%S")
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
        return f"{ts}{rand}"

    cid = corr_id or short_id()
    
    profile = await Profile.find_one({"name": profile_name})
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found")

    run = AnalysisRun(
        symbol=symbol,
        share_name=share_name,
        profile=profile_name,
        model=model,
        corr_id=cid,
        iterations=iters,
        rpm=rpm,
        max_retry=max_retry,
        status="PENDING"
    )
    await run.insert()
    
    # We return the run object. The caller (api.py) will start the background task.
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
