import os
import json
import time
import random
import string
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime
from src.db.models import Profile, AnalysisRun, QuantitativeCriterion
from src.agentic_analysis.agent import NebulaAgent
import pandas as pd
import httpx
import math


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


def _find_metric(data, metric, category=None):
    if isinstance(data, dict):
        if category and category in data and isinstance(data[category], dict):
            if metric in data[category]:
                return data[category][metric]
            for k, v in data[category].items():
                if k.lower() == metric.lower():
                    return v
        if metric in data:
            return data[metric]
        for k, v in data.items():
            if isinstance(k, str) and k.lower() == metric.lower() and not isinstance(v, dict):
                return v
        for v in data.values():
            if isinstance(v, dict):
                result = _find_metric(v, metric)
                if result is not None:
                    return result
    return None


def evaluate_metric(val, criterion: QuantitativeCriterion):
    try:
        operator = criterion.operator
        threshold = criterion.value
        threshold_upper = criterion.value_upper
        metric_type = criterion.metric_type

        if val is None:
            return 0.0

        if metric_type in ("number", "currency", "percentage"):
            val = float(val)
            threshold = float(threshold)
            spread = max(abs(threshold), 1.0) * 0.2

            if operator == "gt":
                if val > threshold:
                    return 1.0
                return sigmoid((val - threshold) / spread)
            elif operator == "gte":
                if val >= threshold:
                    return 1.0
                return sigmoid((val - threshold) / spread)
            elif operator == "lt":
                if val < threshold:
                    return 1.0
                return sigmoid((threshold - val) / spread)
            elif operator == "lte":
                if val <= threshold:
                    return 1.0
                return sigmoid((threshold - val) / spread)
            elif operator == "eq":
                return 1.0 if val == threshold else 0.0
            elif operator == "between":
                if threshold_upper is not None:
                    upper = float(threshold_upper)
                    if threshold <= val <= upper:
                        return 1.0
                    if val < threshold:
                        return sigmoid((val - threshold) / spread)
                    return sigmoid((upper - val) / spread)
                return 0.0

        elif metric_type == "date":
            val_date = datetime.fromisoformat(str(val))
            threshold_date = datetime.fromisoformat(str(threshold))

            if operator == "before":
                return 1.0 if val_date < threshold_date else 0.0
            elif operator == "after":
                return 1.0 if val_date > threshold_date else 0.0
            elif operator == "between":
                if threshold_upper:
                    upper_date = datetime.fromisoformat(str(threshold_upper))
                    return 1.0 if threshold_date <= val_date <= upper_date else 0.0
                return 0.0

        elif metric_type == "text":
            if operator == "eq":
                return 1.0 if str(val).lower() == str(threshold).lower() else 0.0

        return 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


async def fetch_financial_ratios(source: str, symbol: str):
    voyager_base_url = os.getenv("VOYAGER_URL", "http://localhost:8001")
    url = f"{voyager_base_url}/financial-ratios"
    params = {"symbol": symbol, "source": source, "consolidated": "Consolidated"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=15.0)
            if response.status_code == 200:
                raw = response.json()
                records = raw.get("records", [])
                if records:
                    record = records[0]
                    flat = {}
                    for section in ("ratios", "growth"):
                        section_data = record.get(section, {})
                        if isinstance(section_data, dict):
                            for key, val in section_data.items():
                                if isinstance(val, dict):
                                    flat.update(val)
                                else:
                                    flat[key] = val
                    return flat
                return raw
            logger.error(f"Voyager financial-ratios error: {response.status_code}")
        except Exception as e:
            logger.error(f"Error calling Voyager financial-ratios: {str(e)}")
    return None


def _resolve_api_key(model: str, api_keys: dict) -> str:
    provider = model.split("/")[0].lower() if "/" in model else model.lower()
    if provider in ("ollama", "local"):
        return None
    key = api_keys.get(provider) or os.getenv(f"{provider.upper()}_API_KEY")
    return key


async def perform_analysis_task(analysis_id: str, api_keys: dict = None):
    try:
        task_start = time.time()
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

        api_keys = api_keys or {}

        # Quantitative Analysis
        quant_results = {}
        total_weighted_quant_score = 0.0
        total_quant_weight = 0.0

        if run.quantitative and run.source:
            data = await fetch_financial_ratios(run.source, run.symbol)

            if data:
                for criterion in run.quantitative:
                    metric = criterion.metric
                    val = _find_metric(data, metric, criterion.category)

                    score = evaluate_metric(val, criterion)

                    quant_results[metric] = {
                        "value": val,
                        "score": round(score, 4),
                        "weightage": criterion.weightage,
                        "category": criterion.category,
                        "metric_name": criterion.metric_name,
                        "operator": criterion.operator,
                        "threshold": criterion.value
                    }

                    weighted = score * criterion.weightage
                    total_weighted_quant_score += weighted
                    total_quant_weight += criterion.weightage

        final_quant_score = (total_weighted_quant_score / total_quant_weight * 100) if total_quant_weight > 0 else 0.0

        # Qualitative Analysis (LLM-powered)
        qual_results = {}
        total_weighted_qual_score = 0.0
        total_qual_weight = 0.0
        qual_errors = []

        agent_api_key = _resolve_api_key(run.model, api_keys)
        agent = NebulaAgent(model=run.model, api_key=agent_api_key)

        for qual_cfg in run.qualitative:
            param = qual_cfg.parameter
            guidelines = qual_cfg.content
            weight = qual_cfg.weightage
            if not param:
                continue

            param_error = None
            try:
                result = await agent.analyze_parameter(
                    symbol=run.symbol,
                    parameter=param,
                    guidelines=guidelines,
                    documents=run.documents,
                    web_search=run.web_search,
                    web_sources=run.web_sources,
                )
                score = result["score"]
                analysis_text = result["analysis"]
                if result.get("error"):
                    param_error = result["error"]
            except Exception as e:
                logger.error(f"LLM qualitative analysis failed for '{param}': {e}")
                score = 50.0
                analysis_text = f"Analysis failed: {str(e)}"
                param_error = str(e)

            if param_error:
                qual_errors.append(f"{param}: {param_error}")

            qual_results[param] = {
                "analysis": analysis_text,
                "score": score,
                "weightage": weight,
                "error": param_error,
            }

            total_weighted_qual_score += score * weight
            total_qual_weight += weight

        final_qual_score = (total_weighted_qual_score / total_qual_weight) if total_qual_weight > 0 else 0.0

        run.runs["latest_quant"] = quant_results
        run.runs["latest_qual"] = qual_results

        run.quantitative_score = round(final_quant_score, 2)
        run.qualitative_score = round(final_qual_score, 2)

        if final_quant_score > 0 and final_qual_score > 0:
            run.total_score = round((final_quant_score + final_qual_score) / 2, 2)
        else:
            run.total_score = round(final_quant_score or final_qual_score, 2)

        if qual_errors:
            run.error = "; ".join(qual_errors)
            if len(qual_errors) == len(run.qualitative):
                run.status = "FAILED"
                logger.warning(f"Analysis {analysis_id} failed: all qualitative params had errors")
            else:
                run.status = "COMPLETED"
                logger.warning(f"Analysis {analysis_id} completed with {len(qual_errors)} qualitative error(s)")
        else:
            run.status = "COMPLETED"

        run.end_time = time.time()
        run.duration = run.end_time - task_start
        await run.save()
        if run.status == "FAILED":
            logger.error(f"Analysis {analysis_id} finished with status FAILED")
        else:
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
    documents: List[str] = None,
    web_search: bool = False,
    web_sources: List[str] = None,
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
        source=profile.source,
        qualitative=profile.qualitative,
        quantitative=profile.quantitative,
        model=model,
        analysis_id=aid,
        documents=documents or [],
        web_search=web_search,
        web_sources=web_sources or [],
        status="PENDING"
    )
    await run.insert()

    return run


async def get_run_scores_logic(analysis_id: str):
    run = await AnalysisRun.find_one({"analysis_id": analysis_id})
    if not run:
        return None

    return {
        "run": run,
        "dataframe": pd.DataFrame(),
        "qualitative_score": run.qualitative_score
    }


async def save_profile_logic(data: Dict[str, Any]):
    name = data.get("name")
    if not name:
        raise ValueError("Profile name is missing")

    profile = await Profile.find_one({"name": name})
    if profile:
        if "source" in data:
            profile.source = data["source"]
        if "qualitative" in data:
            profile.qualitative = [QualitativeParam(**q) for q in data["qualitative"]]
        if "quantitative" in data:
            profile.quantitative = [QuantitativeCriterion(**q) for q in data["quantitative"]]
        await profile.save()
    else:
        profile = Profile(
            name=name,
            source=data.get("source", ""),
            qualitative=[QualitativeParam(**q) for q in data.get("qualitative", [])],
            quantitative=[QuantitativeCriterion(**q) for q in data.get("quantitative", [])]
        )
        await profile.insert()
    return profile


async def get_available_profiles_logic():
    return await Profile.find_all().to_list()


async def get_profile_logic(name: str):
    return await Profile.find_one({"name": name})


async def get_analysis_runs_logic():
    return await AnalysisRun.find_all().sort(+AnalysisRun.created_at).to_list()


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
