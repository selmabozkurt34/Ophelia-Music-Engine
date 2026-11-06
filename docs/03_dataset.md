# 3. Aşama — Veri Toplama ve Hazırlama

Amaç: model eğitiminde kullanılabilecek, **ticari kullanım hakkı doğrulanmış**,
temizlenmiş ve bölünmüş bir MIDI veri seti.

## 1. Kaynak araştırması ve lisans kararları

11 kaynak değerlendirildi. Kararlar koda gömüldü
([ophelia/data/sources.py](../ophelia/data/sources.py)) ve `ophelia data sources`
komutuyla görülebilir — böylece "hangi veri kullanılabilir" sorusu belgede değil,
**çalışan kodda** yanıtlanır.

| Kaynak | Lisans | Ticari | Karar |
| --- | --- | --- | --- |
| PDMX | CC0-1.0 | evet | ✅ onaylı |
| OpenScore Lieder | CC0-1.0 | evet | ✅ onaylı |
| Bach Chorales (music21) | PD | evet | ✅ onaylı (tohum veri seti) |
| js-fakes | CC-BY-4.0 | evet | ✅ onaylı |
| GiantMIDI-Piano | CC-BY-4.0 | evet | ⚠️ inceleme (disclaimer okunmadı) |
| Lakh (LMD) | CC-BY-4.0 | etiket evet | ❌ reddedildi |
| MAESTRO | CC-BY-NC-SA-4.0 | **hayır** | ❌ reddedildi |
| MetaMIDI | research-only | **hayır** | ❌ reddedildi |
| GigaMIDI | research-only | **hayır** | ❌ reddedildi |
| Essen Folksong | research-only | **hayır** | ❌ reddedildi |
| Nottingham | GPL-3.0 | copyleft | ❌ reddedildi |

### Neden bazı "açık" kaynaklar reddedildi?

- **Lakh (CC BY 4.0):** etiket izin verse de dosyalar internetten toplanmış ve
  telifli popüler şarkılar içeriyor. Lisansı veren kişi eserlerin hak sahibi
  değil; üretken model çıktısı ticari satılacağı için risk kabul edilemez.
- **MAESTRO / MetaMIDI / GigaMIDI:** lisans açıkça NonCommercial veya yalnızca
  araştırma. Model ağırlıkları ve çıktılar ticari olacağı için doğrudan elendi.
- **Nottingham:** GPLv3 bir *yazılım* lisansı; veri için uygun değil ve copyleft
  ticari model için riskli.
- **GiantMIDI:** CC BY 4.0, ancak indirme bir `disclaimer.md` kabulü gerektiriyor
  ve bu metin doğrulanamadı; "review" durumunda bırakıldı.

## 2. Tohum veri seti: Bach koralleri

İlk model (5. aşama) için dar ama **lisans zinciri en net** veri seçildi:

- Eserler J.S. Bach'a ait (ö. 1750) → dünya çapında **kamu malı**
- Kodlamalar BSD-3 lisanslı `music21` paketiyle dağıtılıyor
- Tek stil (4 sesli korale) → küçük model için ideal başlangıç, polifoni ~4

`scripts/fetch_corpus.py` korpusu MIDI'ye çevirir:

```bash
./.venv/Scripts/python.exe scripts/fetch_corpus.py            # 433 eser → 430 MIDI
./.venv/Scripts/python.exe scripts/fetch_corpus.py --limit 10 # deneme
```

## 3. Lisans kayıt sistemi

İki dosya, iki farklı sorumluluk:

**`data/manifests/sources.json`** — kaynak başına lisans ve atıf:

```json
{
  "generated_at": "2026-11-04T21:43:00Z",
  "sources": [
    {
      "slug": "music21-bach-chorales",
      "name": "Bach Chorales (music21 corpus)",
      "license_id": "PD",
      "url": "https://github.com/cuthbertLab/music21",
      "attribution": "Bach Chorales (music21 corpus) (atıf gerekmiyor)",
      "retrieved_at": "2026-11-04T21:46:00Z",
      "file_count": 427,
      "notes": "..."
    }
  ]
}
```

**`data/manifests/files.jsonl`** — dosya başına köken ve bütünlük (satır başına bir JSON):

```json
{"relpath": "raw/music21-bach-chorales/bwv1.6.mid", "sha256": "…",
 "size_bytes": 1234, "source": "music21-bach-chorales",
 "license_id": "PD", "stage": "raw", "added_at": "…"}
```

Kayıt anahtarı `(relpath, stage)`'dir: dosya yeniden üretilirse kayıt **güncellenir**
(yinelenen geçmiş birikmez); farklı yoldaki aynı içerik `DUPLICATE` olarak bildirilir
ve dosya veri setinden çıkarılır.

`sources.json` sürüm kontrolünde tutulur (lisans kaydı kalıcı olmalı).
`files.jsonl` ise üretilen bir defterdir; 427 dosyada ~224 KB olduğu için
`.gitignore`'a alındı ve boru hattı her çalıştığında yeniden üretilir.

Doğrulama:

```bash
ophelia data verify     # eksik/bozuk dosya var mı? (sha256 karşılaştırması)
```

## 4. Boru hattı

```
data/raw/*.mid
      │  1. lisans kapısı     assert_usable(kaynak)   → onaysız kaynak = hata
      │  2. tekilleştirme     sha256                 → aynı içerik atılır
      │  3. temizleme         clean.py               → ret sebepleri kodlanır
      │  4. normalizasyon     normalize.py           → PPQ 480, tek tempo, 16'lık ızgara
      │  5. kayıt             manifest.py            → sha256 + lisans
      │  6. bölütleme         split.py               → train/val/test (kararlı hash)
      ▼  7. rapor             quality.py             → süre, nota, polifoni, tonalite
data/processed/<kaynak>/*.mid + data/splits/splits.json
```

**Temizleme eşikleri** ([clean.py](../ophelia/data/clean.py)): en az 32 nota,
5–900 sn, polifoni ≤ 12, ≥ 4 farklı perde, PPQ ≥ 96, perküsyon yok, 20–300 BPM.

**Normalizasyon** ([normalize.py](../ophelia/data/normalize.py)): hedef PPQ 480,
tek tempo (100 BPM), tek ölçü (4/4), başlangıç ve süre ayrı ayrı 16'lık ızgaraya
oturtulur, kanallar 0'a birleştirilir, perde 21–108 aralığına kırpılır. Çıktı
**deterministiktir** (aynı girdi → aynı baytlar), bu da testlerle doğrulanır.

**Bölütleme** kararlıdır: `sha256(f"{seed}:{dosya_adı}")` → [0,1). Yeni dosya
eklemek eski dosyaların bölümünü değiştirmez.

## 5. Üretilen veri seti

```
Kaynak            : Bach Chorales (music21 corpus) — PD
Ham dosya         : 430 MIDI
Kabul edilen      : 428
Elenen            : 2  (polifoni sınırı aşıldı)
Normalizasyon tekrarı: 1 (bwv197.7-a ≡ bwv398, aynı koral)
Yazılan           : 427
Bölünme           : train 340 · val 49 · test 38
Toplam süre       : 4,56 saat
Toplam nota       : 116.713
Ortalama dosya    : 38,5 sn · 272 nota
Nota yoğunluğu    : 7,1 nota/sn
Ortalama polifoni : 4,21
Ölçü              : %100 4/4
Tonalite (ilk 200): G majör 33 · D majör 23 · A minör 23 · B minör 20 · F majör 17
```

Disk kullanımı: ham 2,2 MB + işlenmiş 1,9 MB.

## 6. Komutlar

```bash
ophelia data sources                  # lisans kararları ve gerekçeleri
ophelia data sources -s approved      # yalnızca kullanılabilir kaynaklar
ophelia data build -s music21-bach-chorales
ophelia data report                   # kalite raporu
ophelia data verify                   # manifest ↔ disk bütünlüğü
```

## 7. Veri setini büyütme (sonraki adım)

5. aşamadaki model için tek stil yeterli olmayacak. Planlanan sıra:

1. **PDMX** (`no_license_conflict` alt kümesi) — en geniş, en temiz lisans; 6.250 saat
2. **OpenScore Lieder** (CC0) — vokal + piyano, MuseScore'dan dönüştürülecek
3. **GiantMIDI** — ancak `disclaimer.md` okunup kabul edildikten sonra

Her yeni kaynak `sources.py`'ye bir satır olarak eklenir; boru hattı aynı kalır.

## 8. Bilinen kısıtlar

- Veri seti **tek stilli** (Bach koralleri, 4 ses, tamamı 4/4). Melodi/ritim
  çeşitliliği için PDMX gerekli.
- Aynı eserin farklı aranjmanları farklı dosyalar olarak girerse bölümler arasında
  sızıntı olabilir; bölütleme şu an dosya adı köküne dayanıyor. `split.py` grup
  eşlemesini destekliyor, boru hattına bağlanması 4. aşamada yapılacak.
- Tonalite tahmini perde sınıfı ağırlığına dayanıyor; minör sonuçlar Bach
  koralleri için fazla temsil ediliyor (bilinen sınır, 2. aşama belgesinde not edildi).
- 2 dosya polifoni eşiği (12) yüzünden elendi; eşik yapılandırılabilir.
