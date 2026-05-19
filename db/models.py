from beanie import Document, Indexed
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import Field

class Profile(Document):
    name: Indexed(str, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional fields from previous iterations if still needed
    llm: Optional[str] = None
    max_rpm: Optional[int] = None
    max_iter: Optional[int] = None
    max_retry_limit: Optional[int] = None
    docs_dir: Optional[str] = None
    documents: Optional[List[str]] = None
    duration: Optional[float] = None

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
    fundamental_analysis: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    runs: Dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0
    end_time: Optional[float] = None

    class Settings:
        name = "analysis_runs"
