import typer
import os
import json
from loguru import logger
from typing import List, Optional
from db.connection import init_db
import asyncio
from functools import wraps
from src.core import (
    create_template_logic,
    save_profile_logic,
    get_available_profiles_logic,
    get_profile_logic,
    get_analysis_runs_logic,
    run_correlation_logic,
    get_run_scores_logic,
    delete_profile_logic
)

app = typer.Typer()

def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@app.command()
@coro
async def delete_profile(name: str):
    await init_db()
    success = await delete_profile_logic(name)
    if success:
        logger.success(f"Profile '{name}' deleted.")
    else:
        logger.error(f"Profile '{name}' not found.")

@app.command()
@coro
async def create_template(
    name: str = typer.Option(..., help="Name of the profile template"),
    sources: List[str] = typer.Option(["screener"], help="List of data sources to include")
):
    filename = await create_template_logic(name, sources)
    logger.success(f"Template created at: {filename}")
    logger.info("Edit this file manually to add/remove data sources and values.")

@app.command()
@coro
async def save_profile(
    file: str = typer.Option(..., help="Path to the edited JSON template")
):
    await init_db()
    if not os.path.exists(file):
        logger.error(f"File not found: {file}")
        return
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        profile = await save_profile_logic(data)
        logger.success(f"Profile '{profile.name}' saved to database.")
    except Exception as e:
        logger.error(f"Failed to save profile: {e}")

@app.command()
@coro
async def available_profiles():
    await init_db()
    profiles = await get_available_profiles_logic()
    if not profiles:
        logger.info("No profiles found.")
        return
    logger.info("Available Profiles:")
    for p in profiles:
        logger.info(f"- {p.name} (Created: {p.created_at.strftime('%Y-%m-%d %H:%M')})")

@app.command()
@coro
async def read_profile(name: str):
    await init_db()
    profile = await get_profile_logic(name)
    if profile:
        logger.info(f"Profile: {profile.name}")
        logger.info(f"Data Sources: {json.dumps(profile.data_sources, indent=2)}")
        logger.info(f"Parameters: {json.dumps(profile.parameters, indent=2)}")
    else:
        logger.error(f"Profile '{name}' not found.")

@app.command()
@coro
async def available_analysis():
    await init_db()
    runs = await get_analysis_runs_logic()
    if not runs:
        logger.info("No analysis runs found.")
        return
    logger.info("Available Analysis Runs:")
    for r in runs:
        duration = f"{r.duration/60:.0f}m {r.duration%60:.0f}s"
        logger.info(f"ID: {r.corr_id} | {r.symbol} | {r.share_name} | {r.created_at.strftime('%Y-%m-%d %H:%M')} | {duration}")

@app.command()
def available_tools():
    from src.tools import CUSTOM_TOOLS_MAP
    logger.info("Available Tools:")
    for tool, desc in CUSTOM_TOOLS_MAP.items():
        logger.info(f"- {tool}: {desc}")

@app.command()
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
    try:
        run = await run_correlation_logic(
            share_name, symbol, profile_name, model, corr_id, iters, rpm, max_retry
        )
        logger.success(f"Correlation analysis complete for {symbol}. Run ID: {run.corr_id}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

@app.command()
@coro
async def read_scores(corr_id: str):
    await init_db()
    results = await get_run_scores_logic(corr_id)
    if not results:
        logger.error("Run not found.")
        return

    logger.info(f"Scores for {results['run'].symbol} ({results['run'].share_name}):")
    logger.info(f"\n{results['dataframe'].to_string()}")
    logger.info(f"Qualitative Score: {results['qualitative_score']}")

if __name__ == "__main__":
    app()
