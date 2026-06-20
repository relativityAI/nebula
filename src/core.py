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
import math
from src.agentic_analysis.agent import NebulaAgent

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
    voyager_base_url = os.getenv("VOYAGER_URL", "http://localhost:8001")
    url = f"{voyager_base_url}/{source}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params={"symbol": symbol}, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            logger.error(f"Voyager API error for {source}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error calling Voyager API for {source}: {str(e)}")
    return None


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def evaluate_filter(value, filter_cfg):
    try:
        val = float(value)
        direction = filter_cfg.get('direction', 'higher')
        
        # Parse inputs safely
        def to_float(k):
            v = filter_cfg.get(k)
            return float(v) if v not in [None, '', 'null'] else None

        t = to_float('threshold')
        l = to_float('lower')
        u = to_float('upper')

        # Fallback: If no threshold is given but bounds are, use the average as threshold
        if t is None and (l is not None or u is not None):
            t = (l + u) / 2 if (l is not None and u is not None) else (l or u)

        # If absolutely no benchmark data, return neutral/zero
        if t is None:
            return 0.0

        # ---- SCENARIO 1: Val is on the "BAD" UPPER side ----
        if u is not None and val > t:
            if val >= u: 
                return 0.01  # Beyond max upper limit = hard penalty
            
            # Scale distance between threshold (0.0) and upper limit (4.4)
            # This ensures val == t gives ~0.99, and val == u gives ~0.01
            width = u - t
            k = 8.8 / width
            return sigmoid(4.4 - k * (val - t))

        # ---- SCENARIO 2: Val is on the "BAD" LOWER side ----
        elif l is not None and val < t:
            if val <= l: 
                return 0.01  # Beyond min lower limit = hard penalty
            
            # Scale distance between threshold and lower limit
            width = t - l
            k = 8.8 / width
            return sigmoid(4.4 - k * (t - val))

        # ---- SCENARIO 3: Val is on the "GOOD" side (No limits breached) ----
        else:
            # If direction is 'lower' and value is below threshold (e.g., PE is 30, threshold 40)
            # OR direction is 'higher' and value is above threshold (e.g., Growth is 25%, threshold 15%)
            # This is exactly what the user wants! Perfect score.
            if direction == 'lower' and val <= t:
                return 1.0
            if direction == 'higher' and val >= t:
                return 1.0
            
            # Standard directional fallback if limits aren't explicitly provided
            width = abs(t) * 0.2 if t != 0 else 1.0
            k = 4.4 / width
            x = (val - t) * k if direction == 'higher' else (t - val) * k
            return sigmoid(x)

    except (ValueError, TypeError, OverflowError):
        return 0.0
    

async def perform_analysis_task(analysis_id: str):
    try:
        run = await AnalysisRun.find_one({"analysis_id": analysis_id})
        if not run:
            logger.error(f"AnalysisRun {analysis_id} not found for background task")
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
        total_weighted_quant_score = 0.0
        total_quant_weight = 0.0

        for source_cfg in run.data_sources:
            source_name = source_cfg.source
            source_weight = getattr(source_cfg, 'weightage', 1.0)
            if not source_name: continue
            
            data = await fetch_voyager_data(source_name, run.symbol)
            if not data:
                quant_results[source_name] = {"error": "No data from source", "score": 0.0}
                continue
            
            source_results = {"metrics": {}}
            source_raw_score = 0.0
            metric_count = 0

            for filter_cfg in source_cfg.filters:
                metric = filter_cfg.metric
                val = data.get(metric)
                if val is None:
                    # Case insensitive check
                    for k, v in data.items():
                        if k.lower() == metric.lower():
                            val = v
                            break
                
                filter_dict = filter_cfg.model_dump()
                score = evaluate_filter(val, filter_dict)
                source_results["metrics"][metric] = {
                    "value": val,
                    "score": round(score, 4)
                }
                source_raw_score += score
                metric_count += 1
            
            if metric_count > 0:
                avg_source_score = source_raw_score / metric_count
                source_results["score"] = round(avg_source_score, 4)
                source_results["weightage"] = source_weight
                
                total_weighted_quant_score += avg_source_score * source_weight
                total_quant_weight += source_weight
            
            quant_results[source_name] = source_results

        # Calculated total quantitative score (weighted)
        final_quant_score = (total_weighted_quant_score / total_quant_weight * 100) if total_quant_weight > 0 else 0.0
        
        # Qualitative (Agentic) Analysis
        qual_results = {}
        total_weighted_qual_score = 0.0
        total_qual_weight = 0.0
        
        agent = NebulaAgent(model=run.model)
        
        for qual_cfg in run.qualitative:
            param = qual_cfg.parameter
            content = qual_cfg.content
            source = getattr(qual_cfg, 'preferred_source', 'Custom Document')
            weight = getattr(qual_cfg, 'weightage', 1.0)
            
            if not param: continue
            
            # Run agent analysis
            analysis_output = await agent.analyze_parameter(
                symbol=run.symbol,
                parameter=param,
                guidelines=content,
                preferred_source=source
            )
            
            score = analysis_output["score"]
            qual_results[param] = {
                "analysis": analysis_output["analysis"],
                "score": score,
                "weightage": weight,
                "preferred_source": source
            }
            
            total_weighted_qual_score += score * weight
            total_qual_weight += weight

        final_qual_score = (total_weighted_qual_score / total_qual_weight) if total_qual_weight > 0 else 0.0

        run.runs["latest_quant"] = quant_results
        run.runs["latest_qual"] = qual_results
        
        run.quantitative_score = round(final_quant_score, 2)
        run.qualitative_score = round(final_qual_score, 2)
        
        # Total score is average of the two pillars (each 0-100)
        # Note: final_quant_score was multiplied by 100 above to be on same scale as qual (0-100)
        if final_quant_score > 0 and final_qual_score > 0:
            run.total_score = round((final_quant_score + final_qual_score) / 2, 2)
        else:
            run.total_score = round(final_quant_score or final_qual_score, 2)
        
        run.status = "COMPLETED"
        run.end_time = time.time()
        run.duration = run.end_time - run.created_at.timestamp()
        await run.save()
        logger.info(f"Analysis {analysis_id} completed successfully")

    except Exception as e:
        logger.exception(f"Error in perform_analysis_task: {str(e)}")
        if run:
            run.status = "FAILED"
            run.error = str(e)
            await run.save()

async def run_analysis_logic(
    share_name: str,
    symbol: str,
    profile_name: str,
    model: str = "cerebras/qwen-3-32b",
    analysis_id: str = None,
    iters: int = 1,
    rpm: int = 2,
    max_retry: int = 3
):
    def short_id(n=2):
        ts = datetime.utcnow().strftime("%y%m%d%H%M%S")
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
        return f"{ts}{rand}"

    aid = analysis_id or short_id()
    
    profile = await Profile.find_one({"name": profile_name})
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found")

    run = AnalysisRun(
        symbol=symbol,
        share_name=share_name,
        profile=profile_name,
        qualitative=profile.qualitative,
        data_sources=profile.data_sources,
        model=model,
        analysis_id=aid,
        iterations=iters,
        rpm=rpm,
        max_retry=max_retry,
        status="PENDING"
    )
    await run.insert()
    
    # We return the run object. The caller (api.py) will start the background task.
    return run

async def get_run_scores_logic(analysis_id: str):
    run = await AnalysisRun.find_one({"analysis_id": analysis_id})
    if not run:
        return None
    
    profile = await Profile.find_one({"name": run.profile})
    
    dfs = []
    for iter_id, iteration in run.runs.items():
        if not isinstance(iteration, dict) or 'parameters' not in iteration:
            continue
        params, scores = [], []
        for p, data in iteration['parameters'].items():
            params.append(p)
            scores.append(data['score'])
        if params:
            dfs.append(pd.DataFrame({'parameter': params, iter_id: scores}).set_index('parameter'))

    if not dfs:
        return {
            "run": run,
            "dataframe": pd.DataFrame(),
            "qualitative_score": 0.0
        }

    df = pd.concat(dfs, axis=1)
    df['avg'] = df.mean(axis=1).astype(int)
    
    # Updated to use data_sources from the run itself
    weights = {}
    for ds in run.data_sources:
        for f in ds.filters:
            weights[f.metric] = 1.0 # Default weight
            
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

async def delete_analysis_logic(analysis_id: str):
    run = await AnalysisRun.find_one({"analysis_id": analysis_id})
    if run:
        await run.delete()
        return True
    return False
