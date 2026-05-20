"""Merkezi yol & secrets çözümleme.

Yerleşim:

    <parent>/
    ├── ifc_all_violeter_viewer_analiyser/   ← bu repo (silinip clone'lanır)
    ├── secrets.txt                          ← API key (kalıcı, repo dışı)
    └── data/                                ← veri (kalıcı, repo dışı)
        ├── violation_pool.sqlite
        ├── ifc_models/{baseline,violated,imports}/
        ├── exports/  docs/  vectorstore/  runs/  logs/

Repo'yu `git pull` ettiğinde veya komple silip yeniden clone'ladığında
`data/` ve `secrets.txt` parent klasörde durduğu için etkilenmez.

Çözümleme (ilk uyan kazanır):
  • Veri:     `IFC_DATA_HOME` env → `<repo>/../data/` (varsayılan, otomatik
              oluşturulur)
  • Secrets:  `IFC_SECRETS_FILE` env → `<repo>/../secrets.txt`
              (yoksa repo içindeki `secrets.txt` da geriye uyum için okunur)

Cross-platform: tüm yollar `pathlib.Path` üzerinden çözülür, macOS / Linux /
Windows fark etmez. Sabit kodlanmış bir mutlak yol yoktur.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROGRAM_ROOT: Path = Path(__file__).resolve().parent
PARENT_ROOT: Path = PROGRAM_ROOT.parent


# ----------------------------- secrets -----------------------------

def _secrets_candidates() -> list[Path]:
    """Aranan sırayla secrets.txt aday yolları."""
    out: list[Path] = []
    env_override = os.getenv("IFC_SECRETS_FILE", "").strip()
    if env_override:
        out.append(Path(env_override).expanduser().resolve())
    out.append(PARENT_ROOT / "secrets.txt")     # asıl yer
    out.append(PROGRAM_ROOT / "secrets.txt")    # repo içi (yedek / dev)
    out.append(PROGRAM_ROOT / ".env")           # geriye dönük uyum
    out.append(PARENT_ROOT / ".env")
    return out


def secrets_file() -> Optional[Path]:
    """Mevcut secrets dosyasının yolu; yoksa None."""
    for c in _secrets_candidates():
        if c.exists() and c.is_file():
            return c
    return None


def load_secrets() -> Optional[Path]:
    """Bulunan ilk secrets dosyasını ortam değişkenlerine yükle. Yolu döndürür."""
    p = secrets_file()
    if p is not None:
        load_dotenv(p, override=False)
    return p


# ----------------------------- data home -----------------------------

def _resolve_data_home() -> Path:
    raw = os.getenv("IFC_DATA_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return PARENT_ROOT / "data"


def data_home() -> Path:
    """Veri kök klasörü; yoksa oluştur."""
    home = _resolve_data_home()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["IFC_DATA_HOME"] = str(home)
    return home


def _sub(name: str) -> Path:
    p = data_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def docs_dir() -> Path:
    return _sub("docs")


def vectorstore_dir() -> Path:
    return _sub("vectorstore")


def ifc_models_dir() -> Path:
    p = _sub("ifc_models")
    for kind in ("baseline", "baseline_uploaded", "violated", "imports"):
        (p / kind).mkdir(parents=True, exist_ok=True)
    return p


def exports_dir() -> Path:
    return _sub("exports")


def ml_runs_dir() -> Path:
    return _sub("runs")


def logs_dir() -> Path:
    return _sub("logs")


def db_path() -> Path:
    return data_home() / "violation_pool.sqlite"


# ----------------------------- status helpers -----------------------------

def secrets_ok() -> bool:
    return secrets_file() is not None and bool(os.getenv("OPENAI_API_KEY", "").strip())


def status() -> dict:
    """UI üst başlığında gösterilecek özet."""
    return {
        "program_root": str(PROGRAM_ROOT),
        "parent_root": str(PARENT_ROOT),
        "data_home": str(data_home()),
        "secrets_file": str(secrets_file()) if secrets_file() else None,
        "openai_api_key": "set" if os.getenv("OPENAI_API_KEY", "").strip() else "missing",
    }


# Modül import edildiğinde bir kez secrets'i yükle ve veri kökünü garantiye al.
load_secrets()
data_home()


# Geriye uyum: bazı eski modüller `DATA_HOME` sabitini import ediyordu.
DATA_HOME: Path = data_home()
