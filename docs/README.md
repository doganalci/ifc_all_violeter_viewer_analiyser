# docs/ — Pipeline Dokümantasyon İndeksi

Bu klasör, bu dönem geliştirilen LLM + GAT pipeline'ı için akademik ve
teknik referans dokümanlarını içerir.

## Hangi dosyayı ne için aç?

| Dosya | Ne işine yarar | Hedef kitle |
|---|---|---|
| **CONTEXT_BRIEF.md** | Tek dosyadan tüm proje contextini al. Yeni Claude oturumuna yapıştır. | Yeni geliştirici / yeni Claude session |
| **PIPELINE_DOCUMENTATION.md** | Dosya + Excel + UI sayfası + deney haritası. "Hangi dosyaya bakayım?" sorularına cevap. | Geliştirici (referans) |
| **THESIS_REPORT_TEMPLATE.md** | Tez izleme komitesi raporu taslağı. Akademik dilde, deneylerin sayısal sonuçlarıyla. | Danışman, komite |
| **NEXT_STEPS.md** | Önceliklendirilmiş TODO listesi (RAG ihlal, GAT karşılaştırma, vs.). | Planlama |
| **GLOSSARY.md** | Terim sözlüğü (TS 9111, IFC entity'leri, GAT, RAG, GUID vs.). | Yeni bakan herkes |

## Hızlı başlangıç

**Pipeline'ı yeni öğreniyorsan:** sırasıyla `GLOSSARY.md` → `CONTEXT_BRIEF.md` →
`PIPELINE_DOCUMENTATION.md`.

**Tez yazıyorsan:** `THESIS_REPORT_TEMPLATE.md` baz al, `NEXT_STEPS.md`'den
mevcut + gelecek ayır.

**Çalışmaya devam edeceksen:** `NEXT_STEPS.md` P1 → RAG ihlal pipeline'ı.

## Repo köküdeki diğer ilgili dokümanlar

- `../README.md` — proje genel kurulum + ana mimari
- `../WINDOWS_SETUP.md` — Windows kurulum rehberi
- `../.env.example` — env var şablonu
- `../requirements.txt` — Python bağımlılıkları
- `../legacy_docs/` — eski alt-projelerin dokümanları (codex1, viewer, analysis)
