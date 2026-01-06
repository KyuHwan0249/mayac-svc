import logging
from logging.handlers import TimedRotatingFileHandler # <--- 이게 바뀝니다
import os
import sys

# 1. 로그 저장 경로
LOG_DIR = "/logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "server.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name="mayac_svc"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    # 2. 포맷 설정
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # -------------------------------------------------------------------------
    # [수정] 날짜별로 파일이 바뀌는 핸들러 (TimedRotatingFileHandler)
    # -------------------------------------------------------------------------
    file_handler = TimedRotatingFileHandler(
        filename=LOG_FILE_PATH,
        when="midnight",    # 자정(00:00)마다 파일 교체
        interval=1,         # 1일 간격
        backupCount=30,     # 30일이 지난 로그는 자동 삭제 (용량 관리)
        encoding="utf-8"
    )
    
    # 저장될 파일명 접미사 설정 (예: server.log.2023-10-27)
    file_handler.suffix = "%Y-%m-%d" 
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 4. 콘솔 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()