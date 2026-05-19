from fastapi import FastAPI, HTTPException
import uvicorn
import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from db.connection import init_db
from src.core import (
    get_available_profiles_logic,
    get_profile_logic,
    run_correlation_logic,
    get_run_scores_logic,
    save_profile_logic,
    delete_profile_logic
)
from __version__ import __version__

# --- Schemas ---

class ProfileSchema(BaseModel):
    name: str
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class CorrelationRequest(BaseModel):
    share_name: str
    symbol: str
    profile_name: str
    model: str = "cerebras/qwen-3-32b"
    iters: int = 1
    rpm: int = 2

# --- App Initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Nebula", version=__version__, lifespan=lifespan)

# --- Endpoints ---

@app.get("/profiles")
async def list_profiles():
    return await get_available_profiles_logic()

@app.get("/profiles/{name}")
async def get_profile(name: str):
    profile = await get_profile_logic(name)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@app.post("/profiles")
async def create_profile(profile: ProfileSchema):
    return await save_profile_logic(profile.model_dump())

@app.delete("/profiles/{name}")
async def delete_profile(name: str):
    success = await delete_profile_logic(name)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"status": "deleted"}

@app.get("/analysis/{corr_id}")
async def get_analysis(corr_id: str):
    results = await get_run_scores_logic(corr_id)
    if not results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {
        "symbol": results["run"].symbol,
        "score": results["qualitative_score"],
        "data": results["run"].runs
    }

@app.post("/correlate")
async def start_correlation(request: CorrelationRequest):
    try:
        run = await run_correlation_logic(
            share_name=request.share_name,
            symbol=request.symbol,
            profile_name=request.profile_name,
            model=request.model,
            iters=request.iters,
            rpm=request.rpm
        )
        return {"status": "success", "corr_id": run.corr_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)