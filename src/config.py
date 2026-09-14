"""路徑常數。

外部根目錄由 repo 根的 paths.txt 宣告（KEY=路徑，一行一鍵；gitignore，範本見 paths.example.txt）。
缺鍵時依 D:\Alz 佈局相對推導：repo 上兩層 = 子主題根（EEG\）、上三層 = 主題根（D:\Alz）。
推導路徑不存在即報錯，避免從錯誤位置靜默讀寫。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATHS_FILE = PROJECT_ROOT / "paths.txt"
_SUBTHEME_ROOT = PROJECT_ROOT.parents[1]
_ALZ_ROOT = PROJECT_ROOT.parents[2]


def _read_paths() -> dict:
    out = {}
    if _PATHS_FILE.exists():
        for line in _PATHS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


_PATHS = _read_paths()


def _path(key: str, default: Path) -> Path:
    p = Path(_PATHS[key]) if key in _PATHS else default
    if key not in _PATHS and not p.exists():
        raise FileNotFoundError(
            f"paths.txt 未宣告 {key}，且推導路徑不存在: {p}" + chr(10) + f"請在 {_PATHS_FILE} 加一行 {key}=<路徑>"
        )
    return p


# 母帶：edf 根（下有 ACS/ NAD/ P/）
EEG_RAW_DIR = _path("EEG_RAW", _SUBTHEME_ROOT / "data" / "edf")
# 受試者表（去識別）；2024 場次表 common\demographics\sessions_2024（與 hospital_A 母體不同，合併另議）
DEMOGRAPHICS_DIR = _path("DEMOGRAPHICS", _ALZ_ROOT / "common" / "demographics" / "sessions_2024")
# 工作區（子主題層）
WORKSPACE_DIR = _path("WORKSPACE", _SUBTHEME_ROOT / "workspace")
