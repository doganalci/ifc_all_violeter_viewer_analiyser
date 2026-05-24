"""Merkezi veri klasörü çözümleme.

Tüm modüller (codex1 çekirdeği, viewer, ml) verilerini tek bir
`IFC_DATA_HOME` altında tutar. Bu modül o yolu bulur, gerekli alt
klasörleri oluşturur ve `.env` dosyasını yükler.

Çözümleme sırası (ilk bulunan kazanır):
  1. `IFC_DATA_HOME` ortam değişkeni
  2. Kardeş klasör: program reposunun bir üstünde `ifc_desktop_doc_dataset/`
  3. `~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset/`
  4. Hata → kullanıcıya kurulum talimatı

CLI argümanı (örn. `--data-home`) script'lerden ayrı geçilir; o değer
bu modül import edilmeden önce `os.environ["IFC_DATA_HOME"]`'a
yazılırsa otomatik etki eder.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROGRAM_ROOT = Path(__file__).resolve().parent
_SIBLING_NAME = "ifc_desktop_doc_dataset"
_FALLBACK = Path.home() / "Desktop" / "doga_full_ifc_prog" / _SIBLING_NAME


def _from_env() -> Optional[Path]:
    raw = os.getenv("IFC_DATA_HOME", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    # Görece yol verildiyse önce program kökü, sonra cwd'ye göre çöz.
    if not p.is_absolute():
        cand1 = (_PROGRAM_ROOT / p).resolve()
        if cand1.exists():
            return cand1
        cand2 = (Path.cwd() / p).resolve()
        if cand2.exists():
            return cand2
        return cand1  # exist etmiyor olsa da bu seçim — create=True ile oluşur
    return p.resolve()


def _from_sibling() -> Optional[Path]:
    """Program kökünün hemen yanındaki olası veri klasörleri.

    Adlar (öncelik sırası): `ifc_desktop_doc_dataset/`, `data/`
    """
    parent = _PROGRAM_ROOT.parent
    for name in (_SIBLING_NAME, "data"):
        cand = parent / name
        if cand.exists():
            return cand
    return None


def _from_desktop_fallback() -> Optional[Path]:
    return _FALLBACK if _FALLBACK.exists() else None


def resolve_data_home(create: bool = True) -> Path:
    """`IFC_DATA_HOME` olarak kullanılacak yolu döndür."""
    home = _from_env() or _from_sibling() or _from_desktop_fallback()
    if home is None:
        raise RuntimeError(
            "IFC veri klasörü bulunamadı. Şu seçeneklerden birini yap:\n"
            "  1) Veri reposunu komşu klasör olarak klonla:\n"
            f"       git clone https://github.com/doganalci/{_SIBLING_NAME}.git "
            f"{_PROGRAM_ROOT.parent}/{_SIBLING_NAME}\n"
            "  2) Veya `IFC_DATA_HOME` env var ile yolu göster:\n"
            "       export IFC_DATA_HOME=/tam/yol/ifc_desktop_doc_dataset\n"
            "  3) Veya script'i `--data-home /tam/yol` ile çalıştır."
        )
    if create:
        home.mkdir(parents=True, exist_ok=True)
    # Sonraki import'lar tutarlı olsun diye env var'ı normalize et.
    os.environ["IFC_DATA_HOME"] = str(home)
    return home


# Çözümleme öncesinde program kökündeki .env'yi yükle — kullanıcının
# `IFC_DATA_HOME=...` satırını oraya yazabilmesi için. Veri klasöründeki
# .env çözümlemeden sonra yüklenir (zaten oraya `OPENAI_API_KEY` vs. girer).
load_dotenv(_PROGRAM_ROOT / ".env", override=False)

# Modül yüklendiğinde bir kez çöz; tüm alt modüller bu değeri paylaşır.
try:
    DATA_HOME: Path = resolve_data_home(create=True)
except RuntimeError:
    # Geç çözümleme: bazı script'ler kendi argümanlarını parse edip
    # sonradan `os.environ["IFC_DATA_HOME"]` set edebilir. O zaman
    # ilk gerçek erişimde çözeriz.
    DATA_HOME = None  # type: ignore[assignment]


def data_home() -> Path:
    """Lazy erişim — DATA_HOME boşsa şimdi çöz."""
    global DATA_HOME
    if DATA_HOME is None:
        DATA_HOME = resolve_data_home(create=True)
    return DATA_HOME


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
    for kind in ("baseline", "violated", "imports"):
        (p / kind).mkdir(parents=True, exist_ok=True)
    return p


def exports_dir() -> Path:
    return _sub("exports")


def ml_runs_dir() -> Path:
    return _sub("ml_runs")


def db_path() -> Path:
    return data_home() / "violation_pool.sqlite"


def env_file() -> Path:
    return data_home() / ".env"


# Veri klasöründeki .env (API anahtarları vs.) — program .env zaten
# yukarıda yüklendi, override etmiyoruz.
if DATA_HOME is not None:
    _data_env = env_file()
    if _data_env.exists():
        load_dotenv(_data_env, override=False)
