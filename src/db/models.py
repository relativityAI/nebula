from beanie import Document, Indexed
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import Field

# class Profile(Document):
#     name: Indexed(str, unique=True)
#     created_at: datetime = Field(default_factory=datetime.utcnow)
#     data_sources: Dict[str, Any] = Field(default_factory=dict)
#     parameters: Dict[str, Any] = Field(default_factory=dict)
    
#     # Metadata
#     llm: Optional[str] = None
#     max_rpm: Optional[int] = None
#     max_iter: Optional[int] = None
#     max_retry_limit: Optional[int] = None

#     class Settings:
#         name = "profiles"


from pydantic import BaseModel, Field

class QualitativeModel(BaseModel):
    parameter: str
    content: str
    # tools: Optional[List[Tool]] = []
    # weight: Optional[float] = 0.0

class DataSourceFilter(BaseModel):
    metric: str
    direction: Optional[Any] = "higher"
    threshold: Optional[Any] = None
    upper: Optional[Any] = None
    lower: Optional[Any] = None
    title: Optional[str] = None
    type: Optional[str] = None

class DataSourceFilter(BaseModel):
    metric: str
    direction: Optional[Any] = "higher"
    threshold: Optional[Any] = None
    upper: Optional[Any] = None
    lower: Optional[Any] = None
    title: Optional[str] = None
    type: Optional[str] = None

class DataSourceModel(BaseModel):
    source: str
    image: Optional[str] = ""
    filters: List[DataSourceFilter] = []

class Profile(Document):
    name: Indexed(str, unique=True)
    qualitative: List[QualitativeModel]
    data_sources: List[DataSourceModel]
    created_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "profiles"

class AnalysisRun(Document):
    corr_id: Indexed(str, unique=True)
    symbol: str
    share_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    profile: str
    model: str
    iterations: int
    rpm: int
    max_retry: int
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    fundamental_analysis: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    quantitative_analysis: Optional[Dict[str, Any]] = None
    total_score: float = 0.0
    error: Optional[str] = None
    runs: Dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0
    end_time: Optional[float] = None

    class Settings:
        name = "analysis_runs"
