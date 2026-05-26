"""Deney & Veri Kümesi Defteri — data klasöründeki Excel kayıtları.

  * experiments.xlsx → her eğitim bir satır (hiperparametre + metrik + paket).
  * datasets.xlsx    → paketler + her örneğin soy ağacı (ana baseline/baseline).

Eğitimler otomatik kaydedilir (GAT Eğitim sonunda). Veri kümesi defteri bu
sayfadaki butonla DB'den yeniden üretilir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import folder_browser, get_dataset_root, set_dataset_root
from ml import tracking
from paths import data_home

st.set_page_config(page_title="Kayıtlar", layout="wide", page_icon="📒")

st.sidebar.title("IFC Graph Analysis")
_cur = get_dataset_root()
_root = st.sidebar.text_input("Dataset klasörü", value=_cur)
if _root != _cur:
    set_dataset_root(_root)
with st.sidebar.expander("📁 Klasör seç (gez)"):
    _picked = folder_browser(start=_root, key="rec_browser")
    if _picked:
        set_dataset_root(_picked)
        st.rerun()

st.title("📒 Deney & Veri Kümesi Defteri")
home = data_home()
st.caption(f"Kayıtlar şurada tutuluyor: `{home}`")


def _read_any(xlsx: Path, sheet: str | None = None) -> pd.DataFrame | None:
    """xlsx varsa onu, yoksa .csv'yi oku."""
    try:
        if xlsx.exists():
            return pd.read_excel(xlsx, sheet_name=sheet) if sheet else pd.read_excel(xlsx)
    except Exception as e:
        st.warning(f"{xlsx.name} okunamadı: {e}")
    csv = xlsx.with_suffix(".csv")
    if csv.exists():
        try:
            return pd.read_csv(csv)
        except Exception:
            return None
    return None


def _download(path: Path, label: str):
    p = path if path.exists() else path.with_suffix(".csv")
    if p.exists():
        with open(p, "rb") as f:
            st.download_button(label, f, file_name=p.name, key=f"dl_{p.name}")


# ---- Eğitim defteri --------------------------------------------------------
st.subheader("🧪 Eğitim defteri (experiments)")
exp = home / "experiments.xlsx"
df_exp = _read_any(exp, sheet="deneyler") if exp.exists() else _read_any(exp)
if df_exp is not None and not df_exp.empty:
    st.caption(f"{len(df_exp)} eğitim kaydı. En yeni en altta.")
    st.dataframe(df_exp, use_container_width=True, height=420)
    _download(exp, "📥 experiments indir")
    # Hızlı karşılaştırma: test F1/precision/recall
    cols = [c for c in ("run", "test_F1", "test_precision", "test_recall",
                        "paketler (ana baseline)") if c in df_exp.columns]
    if len(cols) >= 2:
        with st.expander("📊 Test metrik karşılaştırması", expanded=True):
            st.dataframe(df_exp[cols], hide_index=True, use_container_width=True)
else:
    st.info("Henüz eğitim kaydı yok. GAT Eğitim'i çalıştırınca otomatik eklenir.")

st.divider()

# ---- Eğitim PDF raporları --------------------------------------------------
st.subheader("📄 Eğitim raporları (PDF)")
st.caption(
    "Her eğitim sonunda otomatik üretilir (runs/<run>/report.pdf + "
    "data/reports/). Aşağıdan seç, indir ya da yeniden üret."
)
_runs_root = Path("runs")
_runs = sorted([p for p in _runs_root.iterdir()
                if p.is_dir() and (p / "summary.json").exists()],
               reverse=True) if _runs_root.exists() else []
if _runs:
    _rsel = st.selectbox("Run", _runs, format_func=lambda p: p.name)
    rc1, rc2 = st.columns(2)
    if rc1.button("🔄 PDF raporu (yeniden) üret"):
        try:
            from ml.report import build_report
            pdf = build_report(_rsel, dataset_root=get_dataset_root())
            if pdf:
                st.success(f"Üretildi: {pdf}")
            else:
                st.error("Rapor üretilemedi (matplotlib kurulu mu?).")
        except Exception as e:
            st.error(f"Hata: {e}")
    _pdf_path = _rsel / "report.pdf"
    if _pdf_path.exists():
        with open(_pdf_path, "rb") as f:
            rc2.download_button("📥 PDF indir", f, file_name=f"{_rsel.name}_report.pdf",
                                mime="application/pdf")
    else:
        rc2.caption("Bu run için henüz PDF yok — soldaki butonla üret.")
else:
    st.caption("Henüz eğitim run'ı yok.")

st.divider()

# ---- Veri kümesi defteri ---------------------------------------------------
st.subheader("📦 Veri kümesi defteri (datasets)")
st.caption(
    "Her paket (ana baseline) ve her örneğin hangi baseline / ana baseline'dan "
    "üretildiği. DB'den üretilir — yeni veri ekledikçe yenile."
)
if st.button("🔄 Veri kümesi defterini DB'den yeniden üret", type="primary"):
    try:
        out = tracking.rebuild_dataset_registry(get_dataset_root(), home=home)
        st.success(f"Yazıldı: {', '.join(str(p) for p in set(out.values()))}")
    except Exception as e:
        st.error(f"Üretilemedi: {e}")

ds = home / "datasets.xlsx"
df_pkg = _read_any(ds, sheet="paketler")
df_smp = _read_any(ds, sheet="ornekler")
if df_pkg is None:
    df_pkg = _read_any(home / "datasets_paketler.xlsx")
if df_smp is None:
    df_smp = _read_any(home / "datasets_ornekler.xlsx")

if df_pkg is not None and not df_pkg.empty:
    st.markdown("**Paketler (ana baseline'lar)**")
    st.dataframe(df_pkg, hide_index=True, use_container_width=True)
if df_smp is not None and not df_smp.empty:
    st.markdown("**Örnekler (soy ağacı)**")
    _pkgs = ["(hepsi)"] + sorted(df_smp["ana_baseline (paket)"].dropna().unique().tolist()) \
        if "ana_baseline (paket)" in df_smp.columns else ["(hepsi)"]
    _f = st.selectbox("Pakete göre süz", _pkgs)
    view = df_smp if _f == "(hepsi)" else df_smp[df_smp["ana_baseline (paket)"] == _f]
    st.dataframe(view, hide_index=True, use_container_width=True, height=420)
    _download(ds, "📥 datasets indir")
elif df_pkg is None:
    st.info("Henüz veri kümesi defteri yok. Yukarıdaki butonla üret.")
