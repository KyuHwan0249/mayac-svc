from pydantic import BaseModel
from typing import Optional, List

class BaseResponse(BaseModel):
    status: str
    message: Optional[str] = None

class ReloadResponse(BaseResponse):
    loaded_count: Optional[int] = None

class RegisterResponse(BaseResponse):
    user_id: str

class SearchResultItem(BaseModel):
    user_id: str
    score: float

class SearchResponse(BaseResponse):
    count: Optional[int] = None
    results: Optional[List[SearchResultItem]] = None

class CompareResponse(BaseResponse):
    score: Optional[float] = None
    # match: Optional[bool] = None

class FaceGroupInfo(BaseModel):
    count: int
    user_ids: List[str]

class SummaryResponse(BaseResponse):
    active: Optional[FaceGroupInfo] = None
    deleted: Optional[FaceGroupInfo] = None
    total_records: Optional[int] = None

class CleanupResponse(BaseResponse):
    deleted_count: Optional[int] = None
