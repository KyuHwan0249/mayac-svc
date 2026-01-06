import ctypes
import os
from ctypes import (
    Structure, POINTER, c_int, c_char_p, c_ubyte, c_bool, byref, Array
)

# =============================================================================
# 1. C++ 구조체(Struct) 정의
#    헤더 파일의 구조와 메모리 레이아웃을 1:1로 매핑합니다.
# =============================================================================

class POINT2D(Structure):
    """
    typedef struct _pt {
        int x;
        int y;
    } POINT2D;
    """
    _fields_ = [
        ("x", c_int),
        ("y", c_int)
    ]

class BBOX(Structure):
    """
    typedef struct _bbox {
        int x;
        int y;
        int width;
        int height;
    } BBOX;
    """
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int)
    ]

class FEATURE_INFO(Structure):
    """
    typedef struct _faceinfo {
        POINT2D l_eye;
        POINT2D r_eye;
        POINT2D nose;
        POINT2D l_lip;
        POINT2D r_lip;
        BBOX bbox;
    } FEATURE_INFO;
    """
    _fields_ = [
        ("l_eye", POINT2D),
        ("r_eye", POINT2D),
        ("nose", POINT2D),
        ("l_lip", POINT2D),
        ("r_lip", POINT2D),
        ("bbox", BBOX)
    ]

# =============================================================================
# 2. SDK 래퍼 클래스
# =============================================================================

class FROneSDK:
    def __init__(self, lib_dir):
        self.lib_dir = lib_dir
        self.lib = None
        self.feature_size = 0
        
        # 라이브러리 로드 (이전 대화에서 해결한 의존성/링크 이슈 해결 로직 포함)
        self._load_library()
        
        # 함수 인자/리턴 타입 설정
        self._set_signatures()
        
        # 초기화 (Config 경로 처리를 위해 작업 디렉토리 임시 변경)
        self._initialize_sdk()

        # Feature 크기 미리 조회
        self.feature_size = self.lib.FROne_FeatureSize()
        print(f"ℹ️ Feature Size: {self.feature_size} bytes")

    def _load_library(self):
        """핵심 라이브러리 강제 순차 로드"""
        from logger import logger
        
        # [핵심] 의존성 순서대로 리스트 작성 (순서 절대 변경 금지)
        # 1. 기반 라이브러리 (OpenCV, Torch)
        # 2. 기능 라이브러리 (Detector, Matcher)
        # 3. 메인 라이브러리 (FROne)
        core_libs = [
            "libopencv_world.so",   
            "libtorch_cuda.so",      # (심볼릭 링크 걸린 파일) FaceDetector가 이걸 찾음
            "libFaceDetector.so",    # <-- 이게 먼저 로드되어야 FROne이 안 죽음
            "libfaceMatcher.so",     
            "libFROne.so"            
        ]
        
        print(f"📚 라이브러리 로드 시작: {self.lib_dir}") # 디버깅용 print

        for lib_name in core_libs:
            lib_path = os.path.join(self.lib_dir, lib_name)
            
            if not os.path.exists(lib_path):
                # 파일이 없으면 로그 남기고 패스 (하지만 필수 파일이 없으면 뒤에서 죽음)
                logger.warning(f"⚠️ 파일 없음(건너뜀): {lib_name}")
                continue

            try:
                # RTLD_GLOBAL: 로드된 심볼을 다른 라이브러리가 갖다 쓸 수 있게 함 (필수)
                logger.info(f"라이브러리 로드 시도: {lib_name}")
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                logger.info(f"  ✅ [Core] 로드 성공: {lib_name}")
            except OSError as e:
                logger.error(f"  ❌ [Core] 로드 실패: {lib_name} -> {e}")
                # 필수 라이브러리 로드 실패 시, 뒤에꺼 해봤자 의미 없으므로 중단
                raise e

        # 마지막으로 Main SDK 핸들 잡기 (위에서 로드했으므로 성공함)
        try:
            logger.info("메인 SDK 라이브러리 핸들 획득 시도: libFROne.so")
            self.lib = ctypes.CDLL(os.path.join(self.lib_dir, "libFROne.so"), mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
             logger.error(f"❌ libFROne 핸들 획득 실패: {e}")
             raise e

    def _initialize_sdk(self):
        """config 폴더 인식을 위해 경로 이동 후 Init"""
        # SDK 구조상 App 폴더에 config가 있다고 가정
        # 실제 경로: /app/FROne_SDK_3.0/3rdparty/sqisoft/lib -> ../../../App
        config_parent_dir = os.path.abspath(os.path.join(self.lib_dir, "../../../App"))
        
        original_cwd = os.getcwd()
        try:
            if os.path.exists(config_parent_dir):
                os.chdir(config_parent_dir)
            
            res = self.lib.FROne_Init()
            if res != 0:
                raise Exception(f"FROne_Init failed code: {res}")
        finally:
            os.chdir(original_cwd)

    def _set_signatures(self):
        """C 함수 시그니처 매핑"""
        # 1. char* FROne_GetVer(void);
        self.lib.FROne_GetVer.restype = c_char_p
        
        # 2. int FROne_Init(void);
        self.lib.FROne_Init.restype = c_int
        
        # 3. int FROne_FeatureSize(void);
        self.lib.FROne_FeatureSize.restype = c_int

        # 4. int FROne_Extract(uc* img, int w, int h, int bit, uc* feat);
        self.lib.FROne_Extract.argtypes = [POINTER(c_ubyte), c_int, c_int, c_int, POINTER(c_ubyte)]
        self.lib.FROne_Extract.restype = c_int

        # 5. int FROne_Match(uc* f1, uc* f2, int* score);
        self.lib.FROne_Match.argtypes = [POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_int)]
        self.lib.FROne_Match.restype = c_int

        # 6. int FROne_Append(uc* feat, int id);
        self.lib.FROne_Append.argtypes = [POINTER(c_ubyte), c_int]
        self.lib.FROne_Append.restype = c_int

        # 7. int FROne_Remove(int id);
        self.lib.FROne_Remove.argtypes = [c_int]
        self.lib.FROne_Remove.restype = c_int

        # 8. int FROne_Identify(uc* feat, int max, int* scores, int* ids);
        self.lib.FROne_Identify.argtypes = [POINTER(c_ubyte), c_int, POINTER(c_int), POINTER(c_int)]
        self.lib.FROne_Identify.restype = c_int

        # 10. int FROne_Coords(uc* img, int w, int h, int bit, FEATURE_INFO* coords);
        self.lib.FROne_Coords.argtypes = [POINTER(c_ubyte), c_int, c_int, c_int, POINTER(FEATURE_INFO)]
        self.lib.FROne_Coords.restype = c_int

        # 11. int FROne_Release(void);
        self.lib.FROne_Release.restype = c_int

    # ================= [ Python Methods ] =================

    def get_version(self) -> str:
        return self.lib.FROne_GetVer().decode('utf-8')

    def release(self):
        return self.lib.FROne_Release()

    def extract_feature(self, img_ptr, w, h, bit=24) -> bytes:
        """이미지 포인터를 받아 특징정보(bytes) 반환"""
        feature_buf = (c_ubyte * self.feature_size)()
        res = self.lib.FROne_Extract(img_ptr, w, h, bit, feature_buf)
        if res != 0: raise ValueError(f"Extract Error: {res}")
        return bytes(feature_buf)

    def match(self, feat1: bytes, feat2: bytes) -> int:
        """두 특징정보 비교 (0~99점)"""
        f1 = (c_ubyte * self.feature_size).from_buffer_copy(feat1)
        f2 = (c_ubyte * self.feature_size).from_buffer_copy(feat2)
        score = c_int(0)
        
        res = self.lib.FROne_Match(f1, f2, byref(score))
        if res != 0: raise ValueError(f"Match Error: {res}")
        return score.value

    def append_feature(self, feat: bytes, user_id: int):
        """1:N 매칭을 위해 메모리에 특징정보 등록"""
        f_ptr = (c_ubyte * self.feature_size).from_buffer_copy(feat)
        res = self.lib.FROne_Append(f_ptr, user_id)
        if res != 0: raise ValueError(f"Append Error: {res}")
        return True

    def remove_feature(self, user_id: int):
        """등록된 특징정보 삭제"""
        res = self.lib.FROne_Remove(user_id)
        if res != 0: raise ValueError(f"Remove Error: {res}")
        return True

    def identify(self, feat: bytes, max_matches=5):
        """1:N 검색 실행"""
        f_ptr = (c_ubyte * self.feature_size).from_buffer_copy(feat)
        
        # 결과 받을 배열 할당
        score_array = (c_int * max_matches)()
        id_array = (c_int * max_matches)()
        
        res = self.lib.FROne_Identify(f_ptr, max_matches, score_array, id_array)
        if res != 0: raise ValueError(f"Identify Error: {res}")
        
        results = []
        for i in range(max_matches):
            if score_array[i] > 0: # 유효한 점수만 리턴
                results.append({"id": id_array[i], "score": round(score_array[i] / 100.0, 1)})
        return results

    def get_coords(self, img_ptr, w, h, bit=24):
        """
        얼굴 랜드마크(눈, 코, 입, 박스) 좌표 반환
        Return: dict (l_eye, r_eye, bbox 등)
        """
        coords = FEATURE_INFO()
        res = self.lib.FROne_Coords(img_ptr, w, h, bit, byref(coords))
        
        if res != 0:
             # 얼굴 미검출 시 예외보다는 None이나 빈 값 리턴이 나을 수 있음
             # raise ValueError(f"Coords Error (Face not found?): {res}")
             return None

        # Python Dict로 변환하여 사용하기 편하게 리턴
        return {
            "l_eye": (coords.l_eye.x, coords.l_eye.y),
            "r_eye": (coords.r_eye.x, coords.r_eye.y),
            "nose":  (coords.nose.x,  coords.nose.y),
            "l_lip": (coords.l_lip.x, coords.l_lip.y),
            "r_lip": (coords.r_lip.x, coords.r_lip.y),
            "bbox":  (coords.bbox.x, coords.bbox.y, coords.bbox.width, coords.bbox.height)
        }

    def reset(self):
        """
        SDK 엔진을 완전히 껐다가 다시 켭니다.
        메모리에 등록된 모든 특징점이 삭제됩니다.
        """
        print("🔄 SDK 리셋 시작 (Release -> Init)...")
        
        # 1. 메모리 해제
        self.lib.FROne_Release()
        
        # 2. 다시 초기화 (기존의 _initialize_sdk 재활용)
        # 이 함수가 config 폴더 경로로 이동해서 Init을 안전하게 수행해줍니다.
        self._initialize_sdk()
        
        print("✅ SDK 리셋 완료. 메모리가 비워졌습니다.")
    # 주의: FROne_Crop은 C++ std::vector를 사용하므로 Python ctypes로 래핑 불가.
    # 대신 Python OpenCV를 사용하세요 (img[y:y+h, x:x+w]).