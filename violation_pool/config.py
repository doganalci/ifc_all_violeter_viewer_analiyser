import os
from dataclasses import dataclass, field
from pathlib import Path

# `paths` modülü `.env` dosyasını yükler ve `IFC_DATA_HOME`'u çözer;
# `os.getenv` okumalarından önce import edilmesi şart.
from paths import (
    db_path as _data_db,
    docs_dir as _data_docs,
    exports_dir as _data_exports,
    ifc_models_dir as _data_ifc,
    vectorstore_dir as _data_vectorstore,
)


def _path_env(key: str, default_factory) -> Path:
    raw = os.getenv(key, "").strip()
    return Path(raw).expanduser().resolve() if raw else default_factory()


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    ifc_llm_model: str = field(
        default_factory=lambda: os.getenv("IFC_LLM_MODEL", "gpt-4o-mini")
    )

    # Yollar IFC_DATA_HOME altında; tekil override hâlâ mümkün.
    data_dir: Path = field(default_factory=lambda: _path_env("DATA_DIR", _data_docs))
    vectorstore_dir: Path = field(
        default_factory=lambda: _path_env("VECTORSTORE_DIR", _data_vectorstore)
    )
    db_path: Path = field(default_factory=lambda: _path_env("DB_PATH", _data_db))
    export_dir: Path = field(default_factory=lambda: _path_env("EXPORT_DIR", _data_exports))
    ifc_dir: Path = field(default_factory=lambda: _path_env("IFC_DIR", _data_ifc))

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.vectorstore_dir, self.export_dir):
            p.mkdir(parents=True, exist_ok=True)
        for kind in ("baseline", "violated", "imports"):
            (self.ifc_dir / kind).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()


METHOD_NAIVE = "naive"
METHOD_OPTIMIZED = "optimized"
METHOD_RAG = "rag"
METHOD_FINETUNE = "finetune"

METHOD_LABELS = {
    METHOD_NAIVE: "1) Tek Promt (LLM, ek işlem yok)",
    METHOD_OPTIMIZED: "2) İyileştirilmiş Tek Promt (LLM)",
    METHOD_RAG: "3) Standart dosyalar + RAG + LLM",
    METHOD_FINETUNE: "4) Standart dosyalar + Fine-tune LLM",
}
