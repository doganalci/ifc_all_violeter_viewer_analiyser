<#
.SYNOPSIS
  Local LLM ortamını kurar: Ollama + qwen2.5:7b-instruct (default).
  Sayfa 21'deki "🏠 Local LLM" akışını ücretsiz çalıştırmak için.

.DESCRIPTION
  Idempotent — birden fazla çalıştırılabilir, yüklü bileşenleri atlar.
    1) Ollama yoksa winget ile kurar (Windows 10/11).
    2) Ollama servisini ayağa kaldırır (zaten çalışıyorsa atlar).
    3) Modeli çeker (zaten varsa atlar).
    4) /v1/chat/completions üzerinden smoke test yapar.

.PARAMETER Model
  Ollama model adı. Default: qwen2.5:7b-instruct (RTX 3050 8 GB'a uyar).
  Alternatif: llama3.1:8b · mistral-nemo:12b-instruct-q4_K_M ·
  qwen2.5:14b-instruct-q4_K_M (yavaş ama kaliteli).

.PARAMETER Endpoint
  Ollama HTTP endpoint'i. Default: http://localhost:11434

.EXAMPLE
  .\scripts\install_local_llm.ps1
.EXAMPLE
  .\scripts\install_local_llm.ps1 -Model llama3.1:8b
#>
[CmdletBinding()]
param(
    [string]$Model = "qwen2.5:7b-instruct",
    [string]$Endpoint = "http://localhost:11434"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "  [X]  $msg" -ForegroundColor Red }

function Test-Endpoint([string]$url, [int]$timeoutSec = 3) {
    try {
        $r = Invoke-WebRequest -Uri "$url/api/version" -UseBasicParsing `
                               -TimeoutSec $timeoutSec
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# --- 1) Ollama yüklü mü? ------------------------------------------------
Write-Step "Ollama kurulu mu?"
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Warn "Ollama bulunamadı. winget ile yüklenecek..."
    try {
        winget install --id Ollama.Ollama --silent `
                       --accept-package-agreements `
                       --accept-source-agreements
    } catch {
        Write-Err "winget ile kurulamadı: $($_.Exception.Message)"
        throw "Manuel kur: https://ollama.com/download"
    }
    # PATH yenile (yeni process'ler için ama bu oturumda da denesin)
    $machinePath = [System.Environment]::GetEnvironmentVariable("PATH","Machine")
    $userPath    = [System.Environment]::GetEnvironmentVariable("PATH","User")
    $env:PATH = "$machinePath;$userPath"
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        # winget bazen PATH'i hemen güncellemez; default kuruluma bak
        $default = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
        if (Test-Path $default) {
            $env:PATH = "$($env:PATH);$env:LOCALAPPDATA\Programs\Ollama"
            $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        }
    }
    if (-not $ollamaCmd) {
        throw "Ollama kuruldu ama PATH'te görünmüyor. Yeni PowerShell aç ve tekrar dene."
    }
}
Write-Ok "Ollama: $($ollamaCmd.Source)"

# --- 2) Servis ayakta mı? -----------------------------------------------
Write-Step "Ollama servisi"
if (Test-Endpoint $Endpoint) {
    Write-Ok "Servis zaten aktif: $Endpoint"
} else {
    Write-Warn "Servis cevap vermiyor. 'ollama serve' arka planda başlatılıyor..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" `
                  -WindowStyle Hidden
    $up = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Endpoint $Endpoint 2) { $up = $true; break }
    }
    if (-not $up) {
        throw "Servis 15 saniye içinde ayağa kalkmadı. 'ollama serve' elle dene."
    }
    Write-Ok "Servis ayağa kalktı."
}

# --- 3) Model var mı? ---------------------------------------------------
Write-Step "Model: $Model"
$installed = (& ollama list) 2>&1 | Out-String
if ($installed -match [regex]::Escape($Model)) {
    Write-Ok "Model zaten yüklü."
} else {
    Write-Warn "Model yok, indiriliyor... (qwen2.5:7b ~4.7 GB, sabırlı ol)"
    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        throw "ollama pull başarısız (exit $LASTEXITCODE)."
    }
    Write-Ok "Model indirildi."
}

# --- 4) Smoke test (OpenAI-uyumlu /v1 endpoint) ------------------------
Write-Step "Smoke test (/v1/chat/completions)"
$payload = @{
    model    = $Model
    messages = @(
        @{ role = "user"; content = "Sadece JSON döndür: {""ok"": true}" }
    )
    stream   = $false
} | ConvertTo-Json -Depth 6 -Compress

try {
    $resp = Invoke-RestMethod -Uri "$Endpoint/v1/chat/completions" `
                              -Method Post -Body $payload `
                              -ContentType "application/json" `
                              -TimeoutSec 120
    $msg = $resp.choices[0].message.content
    if ([string]::IsNullOrWhiteSpace($msg)) {
        Write-Warn "Boş cevap. Yine de servis ayakta, devam edebilirsin."
    } else {
        $preview = $msg.Substring(0, [Math]::Min(160, $msg.Length))
        Write-Ok "Cevap: $preview"
    }
} catch {
    Write-Err "Test çağrısı başarısız: $($_.Exception.Message)"
    throw
}

# --- Özet ---------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host " Local LLM hazir." -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "Streamlit Sayfa 21 ayarları:" -ForegroundColor Yellow
Write-Host "  '🏠 Local LLM kullan' expander → ✅ Aktif"
Write-Host "  Endpoint URL : $Endpoint/v1"
Write-Host "  Model adı    : $Model"
Write-Host ""
Write-Host "İzlemek için (ayrı PowerShell):"
Write-Host "  nvidia-smi -l 2     # GPU + VRAM canlı"
Write-Host "  ollama ps           # yüklü model durumu"
Write-Host ""
