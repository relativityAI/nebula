from fastapi import FastAPI, HTTPException, Request
import uvicorn
import os
import pandas as pd
from pydantic import BaseModel, Field
import json
from loguru import logger
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
from src.db.connection import init_db
from src.db.models import Profile, QualitativeParam, QuantitativeCriterion
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
    source: Optional[str] = None
    qualitative: Optional[List[QualitativeParam]] = None
    quantitative: Optional[List[QuantitativeCriterion]] = None

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "6a15dcadcc56b79582c6a5f9",
                "name": "Quality Growth Profile",
                "source": "NSE",
                "qualitative": [
                    {"parameter": "Management Quality", "content": "Strong board with industry experience", "weightage": 8}
                ],
                "quantitative": [
                    {"category": "raw_income_statement", "metric": "RevenueFromOperations", "metric_name": "Revenue from Operations", "metric_type": "currency", "weightage": 7, "operator": "gt", "value": 10000000, "value_upper": None}
                ]
            }
        }
    }


class ProfileInfo(BaseModel):
    id: PydanticObjectId = Field(..., alias="_id")
    name: str
    source: str
    created_at: datetime
    qualitative: Optional[List[Any]] = None
    quantitative: Optional[List[Any]] = None

    model_config = {"populate_by_name": True}

class AnalysisRequest(BaseModel):
    share_name: str
    symbol: str
    profile_name: str
    model: str = "cerebras/qwen-3-32b"
    documents: List[str] = []
    web_search: bool = False
    web_sources: List[str] = []

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

@app.get("/available-models")
def available_models():
    import litellm
    return litellm.model_list

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


# --- CSV Search Helpers ---

SOURCES_DF = None
STOCKS_DF = None
SEC_STOCKS_DF = None

def get_sources_df():
    global SOURCES_DF
    if SOURCES_DF is None:
        path = "assets/sources.csv"
        if os.path.exists(path):
            SOURCES_DF = pd.read_csv(path, skipinitialspace=True)
        else:
            SOURCES_DF = pd.DataFrame(columns=["NAME", "SYMBOL", "COUNTRY"])
    return SOURCES_DF

def get_stocks_df(source: str = "NSE"):
    if source.upper() == "SEC":
        global SEC_STOCKS_DF
        if SEC_STOCKS_DF is None:
            path = "assets/sec-equities.csv"
            if os.path.exists(path):
                SEC_STOCKS_DF = pd.read_csv(path, skipinitialspace=True)
            else:
                SEC_STOCKS_DF = pd.DataFrame(columns=["ticker", "name", "cik", "exchange"])
        return SEC_STOCKS_DF
    else:
        global STOCKS_DF
        if STOCKS_DF is None:
            path = "assets/nse-equities.csv"
            if os.path.exists(path):
                STOCKS_DF = pd.read_csv(path, skipinitialspace=True)
            else:
                STOCKS_DF = pd.DataFrame(columns=["SYMBOL", "NAME"])
        return STOCKS_DF

@app.get("/available-sources")
async def available_sources():
    df = get_sources_df()
    return df.to_dict(orient="records")

@app.get("/search-stocks")
async def search_stocks(query: str, source: str = "NSE"):
    if not query:
        return []
    df = get_stocks_df(source)
    query = query.lower()
    if source.upper() == "SEC":
        mask = df["name"].astype(str).str.lower().str.contains(query, na=False) | \
               df["ticker"].astype(str).str.lower().str.contains(query, na=False)
    else:
        mask = df["NAME"].astype(str).str.lower().str.contains(query, na=False) | \
               df["SYMBOL"].astype(str).str.lower().str.contains(query, na=False)
    return df[mask].head(50).to_dict(orient="records")


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
        profile = Profile(name=name, source="", qualitative=[], quantitative=[])
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
        
    if profile.name is not None:
        p.name = profile.name
    if profile.source is not None:
        p.source = profile.source
    if profile.qualitative is not None:
        p.qualitative = profile.qualitative
    if profile.quantitative is not None:
        p.quantitative = profile.quantitative
    
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
        "share_name": run.share_name,
        "status": run.status,
        "total_score": run.total_score,
        "quantitative_score": run.quantitative_score,
        "qualitative_score": run.qualitative_score,
        "source": run.source,
        "quantitative_analysis": run.runs.get("latest_quant", {}),
        "qualitative_analysis": run.runs.get("latest_qual", {}),
        "qualitative_tool_calls": run.runs.get("qual_tool_calls", {}),
        "qualitative": [json.loads(q.model_dump_json()) for q in run.qualitative],
        "quantitative": [json.loads(q.model_dump_json()) for q in run.quantitative],
        "error": run.error,
        "created_at": run.created_at,
        "end_time": run.end_time,
        "duration": run.duration,
        "analysis_id": run.analysis_id,
        "model": run.model,
        "documents": run.documents,
        "web_search": run.web_search,
        "web_sources": run.web_sources
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
async def run_analysis_endpoint(body: AnalysisRequest, http_request: Request, background_tasks: BackgroundTasks):
    try:
        api_keys = _extract_api_keys(http_request)
        logger.info(f"Requested model: {body.model}")
        logger.info(f"API keys found in headers: {list(api_keys.keys())}")
        _validate_api_key_for_model(body.model, api_keys)

        run = await run_analysis_logic(
            share_name=body.share_name,
            symbol=body.symbol,
            profile_name=body.profile_name,
            model=body.model,
            documents=body.documents,
            web_search=body.web_search,
            web_sources=body.web_sources,
        )
        background_tasks.add_task(perform_analysis_task, run.analysis_id, api_keys)
        return {"status": "success", "analysis_id": run.analysis_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extract_api_keys(request: Request) -> dict:
    headers = request.headers
    mapping = {
        "openai": "X-LLM-OpenAI-Key",
        "gemini": "X-LLM-Gemini-Key",
        "cerebras": "X-LLM-Cerebras-Key",
        "groq": "X-LLM-Groq-Key",
        "tavily": "X-LLM-Tavily-Key",
    }
    keys = {}
    for provider, header in mapping.items():
        val = headers.get(header)
        if val:
            keys[provider] = val
        else:
            logger.info(f"{header} not provided in request headers")
    return keys


def _validate_api_key_for_model(model: str, api_keys: dict):
    provider = model.split("/")[0].lower() if "/" in model else model.lower()
    logger.info(f"Validating API key for provider '{provider}' (model: {model})")
    if provider in ("ollama", "local"):
        logger.info(f"No API key needed for provider '{provider}'")
        return
    key = api_keys.get(provider) or os.getenv(f"{provider.upper()}_API_KEY")
    if key:
        logger.info(f"API key found for provider '{provider}'")
    else:
        logger.warning(f"No API key found for provider '{provider}' — checking fallbacks")
        env_var = f"{provider.upper()}_API_KEY"
        header_var = f"X-LLM-{provider.capitalize()}-Key"
        logger.warning(f"Checked header '{header_var}' and env var '{env_var}' — neither was set")
        raise HTTPException(
            status_code=400,
            detail=f"No API key available for provider '{provider}'. "
                   f"Provide {header_var} header or set {env_var} env var."
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)