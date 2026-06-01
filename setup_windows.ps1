# ============================================================
# Windows kurulum yardımcısı — torch (GPU/CPU) + bağımlılıklar
# Kullanım (PowerShell):
#   conda activate violation-pool
#   .\setup_windows.ps1
# ============================================================

Write-Host "=== IFC Analyser kurulum ===" -ForegroundColor Cyan

# 1) GPU var mı kontrol
$gpu = $false
try {
    nvidia-smi | Out-Null
    if ($LASTEXITCODE -eq 0) { $gpu = $true }
} catch { $gpu = $false }

if ($gpu) {
    Write-Host "🟢 NVIDIA GPU bulundu — CUDA torch kuruluyor (conda)..." -ForegroundColor Green
    conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
} else {
    Write-Host "⚪ GPU yok — CPU torch kuruluyor..." -ForegroundColor Yellow
    pip install torch --index-url https://download.pytorch.org/whl/cpu
}

# 2) torch_geometric (torch'tan sonra)
Write-Host "torch_geometric kuruluyor..." -ForegroundColor Cyan
pip install torch_geometric

# 3) geri kalan bağımlılıklar
Write-Host "Diğer bağımlılıklar kuruluyor..." -ForegroundColor Cyan
pip install -r requirements.txt
pip install watchdog

# 4) doğrulama
Write-Host "`n=== Doğrulama ===" -ForegroundColor Cyan
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

Write-Host "`n✅ Kurulum bitti. Çalıştır: streamlit run app.py --server.address 0.0.0.0" -ForegroundColor Green
