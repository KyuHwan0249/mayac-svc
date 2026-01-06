import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

def get_secret(secret_name, default=None):
    """
    Docker Secret 파일에서 값을 읽어옵니다.
    파일이 없으면 환경변수에서 읽거나 default를 반환합니다.
    """
    secret_path = f"/run/secrets/{secret_name}"
    try:
        with open(secret_path, "r") as f:
            return f.read().strip()
    except IOError:
        # Secret 파일이 없으면 환경변수(기존 방식) 시도
        return os.getenv(secret_name.upper(), default)

# 1. 환경변수에서 DB 접속 정보 가져오기
# (docker-compose.yml의 environment 섹션에서 설정한 값들을 읽습니다)
DB_USER = get_secret("db_user_id", "mayac_user")
DB_PASSWORD = get_secret("db_user_password", "userpassword")
DB_HOST = os.getenv("DB_HOST", "db") # docker service name
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = get_secret("db_name", "mayac_db")

# DB URL 생성
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 2. 엔진 생성 (Connection Pool 설정)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False
)

# 3. 세션 생성기
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. 모델 부모 클래스
Base = declarative_base()

# 5. DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()