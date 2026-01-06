# db/models.py
from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Text, Float, func
from db.database import Base

class TbFeature(Base):
    __tablename__ = "TB_FEATURE"
    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_data = Column(LargeBinary, nullable=False)
    deleted_yn = Column(String(1), default="N", nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    deleted_at = Column(DateTime, nullable=True)

# [추가] API 로그 테이블
class TbApiLog(Base):
    __tablename__ = "TB_API_LOG"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_method = Column(String(10), nullable=False)     # GET, POST, ...
    request_url = Column(String(255), nullable=False)       # /search, /register ...
    client_ip = Column(String(50), nullable=True)           # 요청한 사람 IP
    
    # 요청 Body는 파일 업로드일 경우 너무 크므로 텍스트만 저장하거나 생략
    request_params = Column(Text, nullable=True)            
    
    response_body = Column(Text, nullable=True)             # 결과 JSON
    status_code = Column(Integer, nullable=False)           # 200, 400, 500
    process_time = Column(Float, nullable=False)            # 처리 소요 시간(초)
    created_at = Column(DateTime, default=func.now(), nullable=False)