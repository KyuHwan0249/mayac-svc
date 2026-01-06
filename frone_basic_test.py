import os
import ctypes
from ctypes import c_int, c_char_p

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_ROOT = os.path.join(BASE_DIR, "FROne_SDK_3.0")
LIB_DIR = os.path.join(SDK_ROOT, "3rdparty", "sqisoft", "lib")

print("BASE_DIR:", BASE_DIR)
print("SDK_ROOT:", SDK_ROOT)
print("LIB_DIR :", LIB_DIR)

# 라이브러리 폴더 안에 뭐 있는지 한 번 찍어 보기
try:
    files = os.listdir(LIB_DIR)
    print("LIB_DIR files:", files)
except FileNotFoundError:
    print("LIB_DIR not found!")
    raise

# 1) 의존 .so들을 먼저 RTLD_GLOBAL로 로딩 (순서 상관없이 대충 다 로드)
for name in files:
    if name.endswith(".so") or ".so." in name:
        path = os.path.join(LIB_DIR, name)
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            print("[OK] loaded dep:", name)
        except OSError as e:
            print("[WARN] failed dep:", name, "=>", e)

# 2) 핵심 FROne 라이브러리 로드
fro_path = os.path.join(LIB_DIR, "libFROne.so")
print("Loading libFROne.so from:", fro_path)
fro = ctypes.CDLL(fro_path, mode=ctypes.RTLD_GLOBAL)
print("[OK] libFROne.so loaded")

# 3) C 함수 시그니처 정의 (PDF 기준)
fro.FROne_GetVer.restype = c_char_p
fro.FROne_GetVer.argtypes = []

fro.FROne_Init.restype = c_int
fro.FROne_Init.argtypes = []

fro.FROne_FeatureSize.restype = c_int
fro.FROne_FeatureSize.argtypes = []

fro.FROne_Release.restype = c_int
fro.FROne_Release.argtypes = []

# 4) 실제 호출 테스트
ver = fro.FROne_GetVer()
print("FROne_GetVer ->", ver.decode("utf-8"))

ret_init = fro.FROne_Init()
print("FROne_Init ->", ret_init)

feature_size = fro.FROne_FeatureSize()
print("FROne_FeatureSize ->", feature_size)

ret_rel = fro.FROne_Release()
print("FROne_Release ->", ret_rel)
