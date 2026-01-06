FROM ubuntu:24.04

# 2. 환경 변수 설정
# - 설치 중 상호작용 방지
# - Python 로그 버퍼링 비활성화 (로그 즉시 출력, print 문 확인용)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. 시스템 패키지 설치
# - python3, pip
# - ffmpeg, libopencv-dev (OpenCV 의존성)
# - build-essential (C/C++ 빌드 도구)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    build-essential \
    libopencv-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 라이브러리 설치
# - FastAPI, Uvicorn (웹 서버)
# - OpenCV (이미지 처리)
# - Pillow, Numpy (데이터 처리)
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    pillow \
    opencv-python \
    python-multipart \
    sqlalchemy \
    pymysql

# 5. 작업 디렉토리 설정
WORKDIR /app

# 6. 환경 변수 설정: 라이브러리 경로 (LD_LIBRARY_PATH)
# - SDK 라이브러리 경로를 시스템이 알 수 있도록 등록
# - 기존 시스템 경로도 유지
# ENV LD_LIBRARY_PATH=/app/FROne_SDK_3.0/3rdparty/sqisoft/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:/app/FROne_SDK_3.0/3rdparty/sqisoft/lib:$LD_LIBRARY_PATH

# 7. 소스 코드 복사
# (docker-compose에서 volume을 쓰더라도, 빌드 시점에 복사해두는 것이 안전함)
# COPY . /app

# 8. 포트 개방
EXPOSE 8080

# 9. 서버 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
