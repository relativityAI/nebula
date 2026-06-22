from beanie import Document, Indexed
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QualitativeParam(BaseModel):
    parameter: str
    content: str
    weightage: int = Field(default=5, ge=1, le=10)


class QuantitativeCriterion(BaseModel):
    category: str
    metric: str
    metric_name: str
    metric_type: str  # "number" | "currency" | "percentage" | "date" | "text"
    weightage: int = Field(default=5, ge=1, le=10)
    operator: str     # "gt" | "gte" | "lt" | "lte" | "eq" | "between" | "before" | "after"
    value: Any
    value_upper: Optional[Any] = None


class Profile(Document):
    name: Indexed(str, unique=True)
    source: str = ""
    qualitative: List[QualitativeParam] = []
    quantitative: List[QuantitativeCriterion] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "profiles"


class AnalysisRun(Document):
    analysis_id: Indexed(str, unique=True)
    symbol: str
    share_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    profile: str
    source: str = ""
    qualitative: List[QualitativeParam] = []
    quantitative: List[QuantitativeCriterion] = []
    model: str
    iterations: int
    rpm: int
    max_retry: int
    status: str = "PENDING"
    total_score: float = 0.0
    quantitative_score: float = 0.0
    qualitative_score: float = 0.0
    error: Optional[str] = None
    runs: Dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0
    end_time: Optional[float] = None

    class Settings:
        name = "analysis_runs"
