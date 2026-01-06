from pydantic import BaseModel
from typing import Optional, List

class BaseResponse(BaseModel):
    status: str
    message: Optional[str] = None

class ReloadResponse(BaseResponse):
    loaded_count: Optional[int] = None

class RegisterResponse(BaseResponse):
    user_id: Optional[int] = None

class SearchResultItem(BaseModel):
    id: int
    score: int

class SearchResponse(BaseResponse):
    count: Optional[int] = None
    results: Optional[List[SearchResultItem]] = None

class CompareResponse(BaseResponse):
    score: Optional[int] = None
    match: Optional[bool] = None

class UserGroupInfo(BaseModel):
    count: int
    ids: List[int]

class SummaryResponse(BaseResponse):
    active: Optional[UserGroupInfo] = None
    deleted: Optional[UserGroupInfo] = None
    total_records: Optional[int] = None

class CleanupResponse(BaseResponse):
    deleted_count: Optional[int] = None