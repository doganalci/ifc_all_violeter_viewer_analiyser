"""LLM-tabanlı kapı ihlal üretimi modülü.

Modüller:
  * door_inject — GPT ile kapı genişliği/yüksekliği değişikliği önerip
    enjekte eden temel fonksiyon.
  * batch — paket bazında uçtan uca batch çalıştırıcı (DB, tam etiketleme,
    defterler dahil).

Etiketler her zaman KURALLA ölçülür (eşik kontrolü); LLM sadece çeşitlilik
sağlar, ground truth dürüst kalır.
"""
