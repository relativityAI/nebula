import typer
import os
import time
import json
import re
import random
import string
import pandas as pd
from loguru import logger
from datetime import datetime
from typing import List, Optional
from db.connection import init_db
from db.models import Profile, AnalysisRun
from src.utils import printmd, read_file
# from src.voyager import nse_financials_analysis
import asyncio
from functools import wraps

app = typer.Typer()

def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@app.command()
@coro
async def create_profile(
    name: str = typer.Option(None),
    file: Optional[str] = typer.Option(None)
):
    await init_db()
    if file:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            # Handle potential nested 'parameters' if the user passes the whole JSON
            profile_data = {
                "name": data.get("name", name),
                "parameters": data.get("parameters", data) # Fallback to entire object if no 'parameters' key
            }
            # Remove 'name' from parameters if it was root
            if "name" in profile_data["parameters"]:
                del profile_data["parameters"]["name"]
                
            profile = Profile(**profile_data)
            await profile.save()
            logger.success(f"Profile '{profile.name}' imported and saved.")
        except Exception as e:
            logger.error(f"Failed to import profile: {e}")
    else:
        if not name:
            name = typer.prompt("Enter profile name")
        profile = Profile(name=name)
        await profile.save()
        logger.success(f"Profile '{name}' created in database.")

@app.command(name="available_profiles")
@coro
async def available_profiles():
    await init_db()
    profiles = await Profile.find_all().to_list()
    if not profiles:
        logger.info("No profiles found.")
        return
    logger.info("Available Profiles:")
    for p in profiles:
        logger.info(f"- {p.name} (Created: {p.created_at.strftime('%Y-%m-%d %H:%M')})")

@app.command(name="read_profile")
@coro
async def read_profile(name: str):
    await init_db()
    profile = await Profile.find_one({"name": name})
    if profile:
        logger.info(f"Profile: {profile.name}")
        logger.info(json.dumps(profile.parameters, indent=2))
    else:
        logger.error(f"Profile '{name}' not found.")

@app.command(name="available_analysis")
@coro
async def available_analysis():
    await init_db()
    runs = await AnalysisRun.find_all().sort(+AnalysisRun.created_at).to_list()
    if not runs:
        logger.info("No analysis runs found.")
        return
    logger.info("Available Analysis Runs:")
    for r in runs:
        duration = f"{r.duration/60:.0f}m {r.duration%60:.0f}s"
        logger.info(f"ID: {r.corr_id} | {r.symbol} | {r.share_name} | {r.created_at.strftime('%Y-%m-%d %H:%M')} | {duration}")

@app.command(name="available_tools")
def available_tools():
    from src.tools import CUSTOM_TOOLS_MAP
    logger.info("Available Tools:")
    for tool, desc in CUSTOM_TOOLS_MAP.items():
        logger.info(f"- {tool}: {desc}")

# @app.command(name="financial_check")
# @coro
# async def financial_check(symbol: str, profile_name: str):
#     await init_db()
#     profile = await Profile.find_one({"name": profile_name})
#     if not profile:
#         logger.error(f"Profile '{profile_name}' not found.")
#         return
    
#     quant = profile.parameters.get('quantitative', {})
#     logger.info(f"Analyzing {symbol} financials...")
#     fundamental = nse_financials_analysis(symbol, thresholds=quant.get('financials'))
#     logger.info(f"Score: {fundamental['score']}")

@app.command(name="correlate_share")
@coro
async def correlate_share(
    share_name: str,
    symbol: str,
    profile_name: str,
    model: str = "cerebras/qwen-3-32b",
    corr_id: str = None,
    iters: int = 1,
    rpm: int = 2,
    max_retry: int = 3
):
    await init_db()
    start = time.time()
    
    from src.technicals import get_price_data, technical_analysis_talib
    from src.agents import NebulAgent

    def short_id(n=2):
        ts = datetime.utcnow().strftime("%y%m%d%H%M%S")
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
        return f"{ts}{rand}"

    def extract_score(text):
        m = re.search(r'FINAL_SCORE:\s*(\d{1,3})', text)
        return int(m.group(1)) if m else None

    corr_id = corr_id or short_id()
    profile = await Profile.find_one({"name": profile_name})
    if not profile:
        logger.error(f"Profile '{profile_name}' not found.")
        return

    run = await AnalysisRun.find_one({"corr_id": corr_id})
    if run:
        logger.info(f"Resuming run {corr_id}...")
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

    qualitative = profile.parameters.get('qualitative', {})
    quantitative = profile.parameters.get('quantitative', {})

    logger.info(f"Starting correlation analysis for {symbol}...")
    
    run.fundamental_analysis = nse_financials_analysis(symbol, thresholds=quantitative.get('financials'))
    price_df = get_price_data(f"{symbol}.NS", "3y")
    run.technical_analysis = technical_analysis_talib(symbol, price_df, thresholds=quantitative.get('technicals'))
    await run.save()

    q = [] # deque not used correctly previously, simple list/timestamp check is fine
    agent = NebulAgent(symbol=symbol, profile=profile_name, model=model)

    for i in range(iters):
        iteration = str(i + 1)
        if iteration not in run.runs:
            run.runs[iteration] = {"iteration": iteration, "start_time": time.time(), "parameters": {}}
        
        for param, config in qualitative.items():
            if param in run.runs[iteration]['parameters']:
                continue

            logger.info(f"Qualitative | {param}")
            
            now = time.time()
            q = [t for t in q if t > now - 60]
            if len(q) >= rpm:
                sleep_time = 60 - (now - q[0])
                time.sleep(sleep_time)
            q.append(time.time())

            response = agent.run(config['title'], config['description'], config['tools'])
            score = extract_score(response.choices[0].message.content)
            
            run.runs[iteration]['parameters'][param] = {
                "score": score,
                "weight": float(config.get('weight', 1.0)),
                "response": response.model_dump()
            }
            run.duration = time.time() - start
            run.end_time = time.time()
            await run.save()

        logger.success(f"Iteration {iteration} complete.")

@app.command(name="read_scores")
@coro
async def read_scores(corr_id: str):
    await init_db()
    run = await AnalysisRun.find_one({"corr_id": corr_id})
    if not run:
        logger.error("Run not found.")
        return

    profile = await Profile.find_one({"name": run.profile})
    f_score = run.fundamental_analysis['score']['composite_score']
    t_score = run.technical_analysis['score']['composite_score']

    dfs = []
    for iter_id, iteration in run.runs.items():
        params, scores = [], []
        for p, data in iteration['parameters'].items():
            params.append(p)
            scores.append(data['score'])
        dfs.append(pd.DataFrame({'parameter': params, iter_id: scores}).set_index('parameter'))

    df = pd.concat(dfs, axis=1)
    df['avg'] = df.mean(axis=1).astype(int)
    
    qual_config = profile.parameters.get('qualitative', {})
    weights = {p: qual_config[p].get('weight', 1.0) for p in qual_config}
    avgs = df['avg']
    qual_score = sum(weights[p] * avgs[p] for p in avgs.index if p in weights) / sum(weights.values())

    logger.info(f"\n{df.to_string()}")
    logger.info(f"Financial Score: {f_score}")
    logger.info(f"Technical Score: {t_score}")
    logger.info(f"Qualitative Score: {round(qual_score, 2)}")

if __name__ == "__main__":
    app()
