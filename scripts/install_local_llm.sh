#!/usr/bin/env bash
# Local LLM kurulumu (Ollama + qwen2.5:7b-instruct). macOS / Linux.
# Sayfa 21 "🏠 Local LLM" akışını ücretsiz çalıştırmak için.
#
# Kullanım:
#   ./scripts/install_local_llm.sh
#   ./scripts/install_local_llm.sh llama3.1:8b
set -euo pipefail

MODEL="${1:-qwen2.5:7b-instruct}"
ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"

step() { printf "\n>> %s\n" "$1"; }
ok()   { printf "  [OK] %s\n" "$1"; }
warn() { printf "  [!]  %s\n" "$1"; }

# 1) Ollama
step "Ollama kurulu mu?"
if ! command -v ollama >/dev/null 2>&1; then
    warn "Ollama yok, kuruluyor (curl | sh)..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
ok "Ollama: $(command -v ollama)"

# 2) Servis
step "Ollama servisi"
if curl -fsS "$ENDPOINT/api/version" >/dev/null 2>&1; then
    ok "Servis zaten aktif: $ENDPOINT"
else
    warn "Servis cevap vermiyor, 'ollama serve' arka planda başlatılıyor..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
        sleep 0.5
        curl -fsS "$ENDPOINT/api/version" >/dev/null 2>&1 && break
    done
    if ! curl -fsS "$ENDPOINT/api/version" >/dev/null 2>&1; then
        echo "  [X] Servis 15 saniye içinde ayağa kalkmadı. 'ollama serve' elle dene." >&2
        exit 1
    fi
    ok "Servis ayağa kalktı."
fi

# 3) Model
step "Model: $MODEL"
if ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
    ok "Model zaten yüklü."
else
    warn "Model yok, indiriliyor... (~4.7 GB)"
    ollama pull "$MODEL"
    ok "Model indirildi."
fi

# 4) Smoke test
step "Smoke test (/v1/chat/completions)"
resp=$(curl -fsS "$ENDPOINT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Sadece JSON döndür: {\\\"ok\\\":true}\"}],\"stream\":false}" \
    || true)
if [ -z "$resp" ]; then
    echo "  [X] Test çağrısı başarısız." >&2
    exit 1
fi
ok "Cevap (ilk 160 char): ${resp:0:160}"

echo
echo "============================================================"
echo " Local LLM hazir."
echo "============================================================"
echo
echo "Streamlit Sayfa 21 ayarları:"
echo "  '🏠 Local LLM kullan' expander → ✅ Aktif"
echo "  Endpoint URL : $ENDPOINT/v1"
echo "  Model adı    : $MODEL"
echo
