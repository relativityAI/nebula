from fastapi import FastAPI, HTTPException
import uvicorn
import os
from pydantic import BaseModel, Field
import json
from loguru import logger
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
from src.db.connection import init_db
from src.db.models import Profile, QualitativeModel, DataSourceModel
from src.core import (
    run_analysis_logic,
    get_run_scores_logic,
    perform_analysis_task,
    get_analysis_runs_logic,
    delete_analysis_logic
)
from datetime import datetime
from beanie import PydanticObjectId

from fastapi import BackgroundTasks
from __version__ import __version__

# --- Schemas ---

class ProfileModel(BaseModel):
    id: Union[PydanticObjectId, str] = Field(..., alias="_id")
    name: Optional[str] = None
    qualitative: Optional[List[QualitativeModel]] = None
    data_sources: Optional[List[DataSourceModel]] = None

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "6a15dcadcc56b79582c6a5f9",
                "name": "Profile Name",
                "qualitative": [],
                "data_sources": []
            }
        }
    }


class ProfileInfo(BaseModel):
    id: PydanticObjectId = Field(..., alias="_id")
    name: str
    created_at: datetime
    qualitative: Optional[List[Any]] = None
    data_sources: Optional[List[Any]] = None

    model_config = {"populate_by_name": True}

class AnalysisRequest(BaseModel):
    share_name: str
    symbol: str
    profile_name: str
    model: str = "cerebras/qwen-3-32b"
    iters: int = 1
    rpm: int = 2

class AnalysisModel(BaseModel):
    analysis_id: str

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "analysis_id": "240529123456ab"
            }
        }
    }

# --- App Initialization ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Nebula", version=__version__, lifespan=lifespan)

# --- Endpoints ---

@app.get("/")
def ping():
    return {"ok": 1}

@app.get("/search-profiles")
async def search_profiles(query: str):
    if not query:
        return []

    matches = (
        await Profile.find(Profile.name == {"$regex": query, "$options": "i"})
        .project(ProfileInfo)
        .sort(-Profile.name)
        .to_list()
    )
    return matches


@app.get("/list-profiles")
async def list_profiles():
    # Use model_dump_json to ensure every single field in the document (including lists)
    # is included in the output, bypassing any default FastAPI/Beanie filtering.
    profiles = await Profile.find_all().to_list()
    return [json.loads(p.model_dump_json()) for p in profiles]


@app.get("/read-profile")
async def read_profile(id: str):
    profile = await Profile.get(id)
    if not profile:
        return None
    return json.loads(profile.model_dump_json())


@app.get("/create-profile")
async def create_profile(
    name: Optional[str] = None
):
    if not name:
        name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
    data = {"name": name, "id": "", "_id": "", "ok": 0, "created_at": ""}
    try:
        profile = Profile(name=name, qualitative=[], data_sources=[])
        await profile.insert()

        data["id"] = str(profile.id)
        data["_id"] = str(profile.id)
        data["ok"] = 1
        data["created_at"] = profile.created_at
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        data["error"] = str(e)
    return data


@app.post("/update-profile")
async def update_profile(profile: ProfileModel):
    p = await Profile.get(profile.id)
    if not p:
        return {"ok": 0, "error": "Profile not found"}
        
    if profile.name:
        p.name = profile.name
    if profile.qualitative is not None:
        p.qualitative = profile.qualitative
    if profile.data_sources is not None:
        p.data_sources = profile.data_sources
    
    await p.save()
    return {"ok": 1}


@app.post("/delete-profile")
async def delete_profile(profile: ProfileModel):
    try:
        p = await Profile.get(profile.id)
        if p:
            await p.delete()
            return {"ok": 1}
        return {"ok": 0, "error": "Profile not found"}
    except Exception as e:
        logger.error(f"Error deleting profile: {e}")
        return {"ok": 0, "error": str(e)}


# --- Analysis Endpoints ---

@app.get("/list-analysis")
async def list_analysis():
    runs = await get_analysis_runs_logic()
    return [json.loads(r.model_dump_json()) for r in runs]


@app.get("/read-analysis")
async def read_analysis(id: str):
    results = await get_run_scores_logic(id)
    if not results:
        return None
    
    run = results["run"]
    return {
        "symbol": run.symbol,
        "status": run.status,
        "total_score": run.total_score,
        "quantitative_analysis": run.quantitative_analysis,
        "qualitative_score": results.get("qualitative_score", 0),
        "error": run.error,
        "created_at": run.created_at,
        "end_time": run.end_time,
        "duration": run.duration,
        "analysis_id": run.analysis_id
    }


@app.post("/delete-analysis")
async def delete_analysis(analysis: AnalysisModel):
    try:
        success = await delete_analysis_logic(analysis.analysis_id)
        if success:
            return {"ok": 1}
        return {"ok": 0, "error": "Analysis not found"}
    except Exception as e:
        logger.error(f"Error deleting analysis: {e}")
        return {"ok": 0, "error": str(e)}


@app.get("/analysis/{analysis_id}")
async def get_analysis_legacy(analysis_id: str):
    return await read_analysis(analysis_id)


@app.post("/run-analysis")
async def run_analysis_endpoint(request: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        run = await run_analysis_logic(
            share_name=request.share_name,
            symbol=request.symbol,
            profile_name=request.profile_name,
            model=request.model,
            iters=request.iters,
            rpm=request.rpm
        )
        background_tasks.add_task(perform_analysis_task, run.analysis_id)
        return {"status": "success", "analysis_id": run.analysis_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)