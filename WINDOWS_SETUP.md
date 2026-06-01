# Windows Kurulum + Uzaktan Erişim

`ifc_all_violeter_viewer_analiyser` projesini Windows üzerinde çalıştırma
ve Mac/başka bir bilgisayardan tarayıcı ile uzaktan kontrol etme rehberi.

## 0) Ön Koşullar

Windows makinede:
- **Anaconda** veya **Miniconda** — https://docs.conda.io/en/latest/miniconda.html
- **Git for Windows** — https://git-scm.com/download/win
- **PowerShell** (Windows ile gelir)

## 1) Tek Komutta Kurulum

PowerShell aç:

```powershell
cd $env:USERPROFILE\Desktop
mkdir ifc_claude_code_directory -ErrorAction SilentlyContinue
cd ifc_claude_code_directory
git clone https://github.com/doganalci/ifc_all_violeter_viewer_analiyser.git
cd ifc_all_violeter_viewer_analiyser
git checkout claude/consolidate-developments-CrLud
conda create -n violation-pool python=3.11 -y
conda activate violation-pool
```

### torch + bağımlılıklar — kurulum script'i (ÖNERİLEN)

```powershell
.\setup_windows.ps1
```

Bu script GPU'yu otomatik algılar:
- **NVIDIA GPU varsa** → `conda install pytorch pytorch-cuda=12.1` (CUDA build)
- **GPU yoksa** → CPU torch
- Sonra torch_geometric + requirements.txt + doğrulama

> ⚠️ **ÖNEMLİ:** Düz `pip install torch` Windows'ta **CPU-only** build verir
> (GPU çalışmaz). Mutlaka script'i veya aşağıdaki manuel adımları kullan.

### Manuel (script çalışmazsa)

```powershell
# GPU varsa (conda, en sağlam):
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
# conda/pip karışırsa: yukarıdakine --force-reinstall ekle

# GPU yoksa:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install torch_geometric
pip install -r requirements.txt
pip install watchdog

# Kontrol:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2) Data Klasörü

`data` klasörünü Mac'inden Windows'a kopyala (en hızlı yöntem):
- **Windows Share / SMB**: Mac'te "System Settings → General → Sharing → File Sharing" aç
  Windows'tan `\\<mac-ip>\<sharedfolder>` ile erişip kopyala
- **USB**: `data/` klasörünü USB'ye at, Windows'a kopyala
- **rsync**: `rsync -avh /Users/doganalci/Desktop/ifc_claude_code_directory/data/ <kullanıcı>@<windows-ip>:/Users/<user>/Desktop/ifc_claude_code_directory/data/`
- **OneDrive / Google Drive / Dropbox**: data klasörünü senkronize et

Sonuçta Windows'ta aynı yerleşim olmalı:
```
C:\Users\<sen>\Desktop\ifc_claude_code_directory\
├── ifc_all_violeter_viewer_analiyser\   (kod)
└── data\                                 (veri — IFC_DATA_HOME otomatik bulur)
    ├── violation_pool.sqlite
    ├── ifc_models\
    ├── docs\
    ├── vectorstore\
    └── .env                              (OPENAI_API_KEY=...)
```

`paths.py` Windows yollarını da otomatik anlar — ek ayar gerekmez.

## 3) API Anahtarı

`data\.env` dosyasında OPENAI_API_KEY=... satırı olmalı.

Yoksa oluştur:
```powershell
notepad ..\data\.env
```
İçine yaz:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
IFC_LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

## 4) Network Erişimine Açık Çalıştır

Mac'inden tarayıcı ile bağlanabilmen için `--server.address 0.0.0.0`:

```powershell
conda activate violation-pool
cd $env:USERPROFILE\Desktop\ifc_claude_code_directory\ifc_all_violeter_viewer_analiyser
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Konsolda göreceğin satır:
```
  Network URL:  http://192.168.1.42:8501
```

**Windows Firewall** çıkışta sorabilir: "Allow access" diyip izin ver.

## 5) Mac'ten Bağlan

Mac'in safari/chrome'unda:
```
http://<windows-ip>:8501
```

`<windows-ip>` yukarıdaki "Network URL"deki IP. Aynı router'daysanız direkt çalışır.

### Farklı Ağdaysanız

**Seçenek A — Tailscale (en kolay, ücretsiz):**
1. Her iki cihaza Tailscale kur: https://tailscale.com/download
2. İkisinde aynı hesapla giriş yap
3. Windows'un Tailscale IP'siyle bağlan: `http://100.x.x.x:8501`

**Seçenek B — ngrok:**
```powershell
ngrok http 8501
```
Verdiği `https://abc123.ngrok.io` adresi public.

**Seçenek C — SSH Port Forwarding** (Mac'ten):
```bash
ssh -L 8501:localhost:8501 <user>@<windows-ip>
# sonra Mac browser'da: http://localhost:8501
```

## 6) Paralel LLM Enjeksiyon

Yeni eklenen paralel iş parçacığı ile **8-16x hızlanma**:

**Toplu enjeksiyon (İhlal enjekte et sekmesi):**
- `🔁 Varyant/baseline: 5`
- `🎲 İhlal/varyant: 5`
- `⚡ Eşzamanlı (paralel): 8`
- **🚀 PAKETİN TÜMÜNE enjekte et** → 250 IFC ~3-5 dakikada

**Otomatik Dataset pipeline:**
- `⚡ Paralel LLM iş parçacığı: 8` (yeni)

> **Not:** OpenAI rate limit'e dikkat — `gpt-4o-mini` için ~500 RPM. 16'dan fazla concurrent yaparsan 429 hatası alabilirsin.

## 7) Eğitim — GPU Kullanımı

NVIDIA kartın varsa GAT Eğitim sayfasında **Cihaz: cuda** seç. CPU'ya göre 10-50x hızlı.

## 8) Sonraki Güncellemeler

```powershell
cd $env:USERPROFILE\Desktop\ifc_claude_code_directory\ifc_all_violeter_viewer_analiyser
git pull
pip install -r requirements.txt
```

## 9) Background Çalıştırma

Streamlit'i terminal kapalıyken devam ettirmek için:

```powershell
# Detached process
Start-Process powershell -ArgumentList "-NoExit", "-Command", "conda activate violation-pool; streamlit run app.py --server.address 0.0.0.0"
```

Veya **NSSM** ile Windows Service olarak kayıt et (kalıcı).

## 10) Sorun Giderme

| Hata | Çözüm |
|---|---|
| `OSError: [WinError 10048] Address already in use` | Başka port: `--server.port 8502` |
| `OPENAI_API_KEY not found` | `data\.env` dosyasını kontrol et |
| `ifcopenshell ImportError` | `conda install -c conda-forge ifcopenshell` |
| `torch_geometric error` | torch'u önce kur, sonra PyG |
| Mac'ten bağlanılamıyor | Windows Firewall'da port 8501 açık mı? |
| Yavaş eğitim | GPU varsa cuda seç, yoksa hidden_dim'i düşür |

## 11) Local LLM (opsiyonel — Sayfa 21 için ücretsiz inject)

Sayfa 21'in "🏠 Local LLM" özelliğini kullanmak için Ollama + bir model.
Tek tıkla kurulum:

```powershell
# Repo kökünden:
.\scripts\install_local_llm.ps1
```

Script: Ollama yoksa winget ile kurar → servisi başlatır →
`qwen2.5:7b-instruct` modelini çeker (~4.7 GB) → smoke test yapar.
Idempotent (tekrar çalıştırmak güvenli).

Farklı model için:
```powershell
.\scripts\install_local_llm.ps1 -Model llama3.1:8b
```

Sonra Streamlit'i yeniden başlat → Sayfa 21'de "🏠 Local LLM kullan"
checkbox'ını işaretle.

macOS/Linux karşılığı: `./scripts/install_local_llm.sh`.
