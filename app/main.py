import os
import cv2
import numpy as np
import traceback
import asyncio
import time
from typing import List, Dict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, APIRouter # APIRouter 추가
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from ctypes import POINTER, c_ubyte
from sqlalchemy.orm import Session
from sqlalchemy import func

from db import get_db, engine, Base, TbFeature, TbApiLog
from db.database import SessionLocal 
from frosdk import FROneSDK
from response_model import *
from logger import logger

# 1. FastAPI 앱 생성 (기본 설정)
app = FastAPI(
    title="Face Recognition API",
    description="FROne SDK 기반 얼굴 인식 및 관리 API",
    version="1.0.0",
    docs_url="/face-svc/docs",       # Swagger UI 주소 변경 (/face/docs)
    redoc_url="/face-svc/redoc",     # ReDoc 주소 변경
    openapi_url="/face-svc/openapi.json" # OpenAPI 스키마 주소 변경
)

# 2. 라우터 생성 (여기에 prefix 설정)
router = APIRouter(prefix="/face-svc")

sdk = None
sdk_lock = asyncio.Lock() # Lock 사용 (Semaphore 대신)

# =============================================================================
# 미들웨어 (기존 유지, 경로 체크 부분만 수정)
# =============================================================================
def write_api_log(log_data: dict):
    try:
        db = SessionLocal()
        log_entry = TbApiLog(**log_data)
        db.add(log_entry)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"❌ API 로그 DB 저장 실패: {e}")

async def iterate_in_chunks(content):
    yield content

@app.middleware("http")
async def api_logging_middleware(request: Request, call_next):
    # [수정] Prefix가 붙은 Docs 경로 제외
    if request.url.path.startswith("/face-svc/docs") or \
       request.url.path.startswith("/face-svc/redoc") or \
       request.url.path.startswith("/face-svc/openapi.json"):
        return await call_next(request)
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"🔥 미들웨어 에러: {e}", exc_info=True)
        response = JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    process_time = time.time() - start_time
    
    response_body_bytes = b""
    try:
        if hasattr(response, "body_iterator"):
            body_chunks = [chunk async for chunk in response.body_iterator]
            response_body_bytes = b"".join(body_chunks)
            response.body_iterator = iterate_in_chunks(response_body_bytes)
        else:
            response_body_bytes = response.body
        response_body_str = response_body_bytes.decode("utf-8")
    except Exception as e:
        response_body_str = f"[Body Read Error: {str(e)}]"

    log_data = {
        "request_method": request.method,
        "request_url": str(request.url.path),
        "client_ip": request.client.host if request.client else "unknown",
        "request_params": "File Upload" if "multipart" in request.headers.get("content-type", "") else str(request.query_params),
        "response_body": response_body_str[:2000],
        "status_code": response.status_code,
        "process_time": round(process_time, 4)
    }
    response.background = BackgroundTask(write_api_log, log_data)
    logger.info(f"📡 [{request.method}] {request.url.path} - {response.status_code} ({round(process_time, 3)}s) IP={log_data.get("getclient_ip")}")
    return response

# =============================================================================
# 헬퍼 함수 및 이벤트 (기존 유지)
# =============================================================================
def sync_features_from_db(sdk_instance: FROneSDK):
    db = SessionLocal()
    count = 0
    total = 0
    try:
        active_features = db.query(TbFeature).filter(TbFeature.deleted_yn == 'N').all()
        total = len(active_features)
        logger.info(f"📥 DB 데이터 동기화 시작 (총 {total}건)...")
        for item in active_features:
            try:
                sdk_instance.append_feature(item.feature_data, item.id)
                count += 1
            except Exception as e:
                logger.warning(f"⚠️ ID {item.id} SDK 등록 실패: {e}")
    except Exception as e:
        logger.error(f"❌ DB 조회 중 오류 발생: {e}")
        # raise e  <-- Startup 중단을 막으려면 주석 처리 가능
    finally:
        db.close()
    logger.info(f"📂 동기화 완료: {count}/{total} 성공.")
    return count

async def process_image_to_ptr(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"❌ 이미지 디코딩 실패: {file.filename}")
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")
    h, w, _ = img.shape
    img_contiguous = np.ascontiguousarray(img, dtype=np.uint8)
    ptr = img_contiguous.ctypes.data_as(POINTER(c_ubyte))
    return ptr, w, h

@app.on_event("startup")
def startup_event():
    global sdk
    logger.info("🚀 DB 테이블 확인 및 생성 시도...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ DB 연결 및 테이블 체크 완료.")
    except Exception as e:
        logger.error(f"⚠️ [경고] DB 연결 실패: {e}")
        logger.error("   -> DB 없이 서버를 시작합니다.")

    LIB_PATH = "/app/FROne_SDK_3.0/3rdparty/sqisoft/lib"
    logger.info(f"🚀 SDK 초기화 시작... 경로: {LIB_PATH}")
    
    if not os.path.exists(LIB_PATH):
        logger.critical(f"❌ 라이브러리 경로 없음")
        return 

    try:
        sdk = FROneSDK(LIB_PATH)
        logger.info(f"✅ SDK 로드 성공 (Ver: {sdk.get_version()})")
        try:
            sync_features_from_db(sdk)
        except Exception as e:
            logger.warning(f"⚠️ [경고] 초기 데이터 로딩 실패: {e}")
    except Exception as e:
        logger.error(f"❌ SDK 초기화 실패: {e}")
        logger.error("Traceback:", exc_info=True)

@app.on_event("shutdown")
def shutdown_event():
    if sdk:
        logger.info("🛑 서버 종료: SDK 메모리 해제")
        sdk.release()

# =============================================================================
# 3. API 엔드포인트 (router 사용)
# =============================================================================

# [중요] 모든 @app.post -> @router.post 로 변경

@router.post("/reload", response_model=ReloadResponse)
async def reload_sdk():
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    logger.info("🔄 SDK Reload 요청됨")
    try:
        async with sdk_lock: 
            sdk.reset()
            loaded_count = sync_features_from_db(sdk)
        return {"status": "success", "message": "SDK Reloaded", "loaded_count": loaded_count}
    except Exception as e:
        logger.error(f"❌ Reload 실패: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@router.post("/register", response_model=RegisterResponse)
async def register_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    try:
        ptr, w, h = await process_image_to_ptr(file)
        async with sdk_lock:
            feature_bytes = sdk.extract_feature(ptr, w, h)
        
        new_feature = TbFeature(feature_data=feature_bytes)
        db.add(new_feature)
        db.commit()
        db.refresh(new_feature)
        generated_id = new_feature.id
        
        try:
            async with sdk_lock:
                sdk.append_feature(feature_bytes, generated_id)
            logger.info(f"✅ 사용자 등록 성공: ID {generated_id}")
        except Exception as sdk_err:
            db.delete(new_feature)
            db.commit()
            raise sdk_err
        
        return {"status": "success", "message": "Face registered.", "face_id": generated_id}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Register 오류: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@router.post("/search", response_model=SearchResponse)
async def search_face(file: UploadFile = File(...), max_results: int = Form(5)):
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    try:
        ptr, w, h = await process_image_to_ptr(file)
        async with sdk_lock:
            probe_feat = sdk.extract_feature(ptr, w, h)
            results = sdk.identify(probe_feat, max_matches=max_results)
        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"❌ Search 오류: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@router.delete("/faces/{face_id}", response_model=BaseResponse)
async def delete_face(face_id: int, db: Session = Depends(get_db)):
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    target = db.query(TbFeature).filter(TbFeature.id == face_id, TbFeature.deleted_yn == 'N').first()
    if not target: return {"status": "fail", "message": "Face not found"}
    try:
        try:
            async with sdk_lock: 
                sdk.remove_feature(face_id)
        except: pass 
        target.deleted_yn = 'Y'
        target.deleted_at = func.now()
        db.commit()
        logger.info(f"🗑️ 사용자 삭제(Soft) 완료: ID {face_id}")
        return {"status": "success", "message": f"Face {face_id} deleted."}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Delete 오류: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/compare", response_model=CompareResponse)
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    try:
        ptr1, w1, h1 = await process_image_to_ptr(file1)
        ptr2, w2, h2 = await process_image_to_ptr(file2)
        async with sdk_lock:
            feat1 = sdk.extract_feature(ptr1, w1, h1)
            feat2 = sdk.extract_feature(ptr2, w2, h2)
            score = sdk.match(feat1, feat2)
        return {"status": "success", "score": round(score / 100.0, 1)}
    except Exception as e:
        logger.error(f"❌ Compare 오류: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/faces/summary", response_model=SummaryResponse)
def get_face_summary(db: Session = Depends(get_db)):
    try:
        active_ids = [r.id for r in db.query(TbFeature.id).filter(TbFeature.deleted_yn == 'N').all()]
        deleted_ids = [r.id for r in db.query(TbFeature.id).filter(TbFeature.deleted_yn == 'Y').all()]
        return {
            "status": "success",
            "active": { "count": len(active_ids), "ids": active_ids },
            "deleted": { "count": len(deleted_ids), "ids": deleted_ids },
            "total_records": len(active_ids) + len(deleted_ids)
        }
    except Exception as e:
        logger.error(f"❌ Summary 오류: {e}")
        return {"status": "error", "message": str(e)}

@router.delete("/faces/{face_id}/hard", response_model=BaseResponse)
async def hard_delete_face(face_id: int, db: Session = Depends(get_db)):
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    target = db.query(TbFeature).filter(TbFeature.id == face_id).first()
    if not target: return {"status": "fail", "message": "Face not found"}
    try:
        try:
            async with sdk_lock:  
                sdk.remove_feature(face_id)
        except: pass
        db.delete(target)
        db.commit()
        logger.info(f"🔥 사용자 영구 삭제 완료: ID {face_id}")
        return {"status": "success", "message": f"Face {face_id} permanently deleted."}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Hard Delete 오류: {e}")
        return {"status": "error", "message": str(e)}

@router.delete("/cleanup", response_model=CleanupResponse)
def cleanup_deleted_faces(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(TbFeature).filter(TbFeature.deleted_yn == 'Y').count()
        if deleted_count == 0: return {"status": "success", "message": "No data"}
        db.query(TbFeature).filter(TbFeature.deleted_yn == 'Y').delete(synchronize_session=False)
        db.commit()
        logger.info(f"🧹 Cleanup 완료: {deleted_count}건")
        return {"status": "success", "message": "Cleaned up", "deleted_count": deleted_count}
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Cleanup 오류: {e}")
        return {"status": "error", "message": str(e)}

@router.put("/faces/{face_id}", response_model=BaseResponse)
async def update_face_feature(face_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not sdk: raise HTTPException(500, "SDK Not Initialized")
    target = db.query(TbFeature).filter(TbFeature.id == face_id, TbFeature.deleted_yn == 'N').first()
    if not target: return {"status": "fail", "message": "Active face not found"}
    try:
        ptr, w, h = await process_image_to_ptr(file)
        async with sdk_lock:
            new_feature_bytes = sdk.extract_feature(ptr, w, h)
            try: sdk.remove_feature(face_id)
            except: pass
            sdk.append_feature(new_feature_bytes, face_id)
        target.feature_data = new_feature_bytes
        db.commit()
        logger.info(f"🔄 사용자 업데이트 완료: ID {face_id}")
        return {"status": "success", "message": f"Face {face_id} feature updated."}
    except Exception as e:
        db.rollback()
        try:
            async with sdk_lock:  
                sdk.remove_feature(face_id)
        except: pass
        logger.error(f"❌ Update 오류: {e}")
        return {"status": "error", "message": str(e)}

@router.delete("/reset", response_model=CleanupResponse)
async def reset_faces(db: Session = Depends(get_db)):
    try:
        deleted_count = db.query(TbFeature).count()
        if deleted_count == 0: return {"status": "success", "message": "No data"}
        db.query(TbFeature).delete(synchronize_session=False)
        db.commit()
        async with sdk_lock:
            sdk.reset()
            loaded_count = sync_features_from_db(sdk)
        logger.info(f"🧹 Reset 완료: {deleted_count}건 삭제됨, {loaded_count}건 로드됨")
        return {"status": "success", "message": "Reset completed", "deleted_count": deleted_count, "loaded_count": loaded_count }
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Reset 오류: {e}")
        return {"status": "error", "message": str(e)}

# [마지막] 라우터를 앱에 등록
app.include_router(router)