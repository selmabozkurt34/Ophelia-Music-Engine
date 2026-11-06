# 5. Aşama — Beste Modeli

Projenin **ilk büyük kilometre taşı**: kendi eğittiğimiz model, kendi ürettiği
token'ları dinlenebilir MIDI'ye çeviriyor.

```
token veri seti  →  940.160 parametreli decoder-only transformer  →  dilbilgisi
                                                                     kısıtlı
                                                                     örnekleme
                                                                        ↓
                                              decode_notes → write_midi → .mid
```

## 1. Mimari

Decoder-only transformer (GPT düzeni), tamamen açık yazılmış
([transformer.py](../ophelia/model/transformer.py)) — hazır blok yerine kendi
bloklarımız, çünkü causal maskenin doğruluğu testlerle kanıtlanmalı.

| Bileşen | Değer |
| --- | --- |
| Katman | 3 × (LN → causal self-attention → LN → MLP), pre-LN, artık bağlantılı |
| d_model / başlık | 128 / 4 (başlık boyu 32) |
| Bağlam | 256 token |
| Gömme | Token + pozisyon; `lm_head` token gömülmesiyle **ağırlık paylaşımlı** |
| Dropout | 0,25 |
| Parametre | **940.160** (formül `ModelConfig.parameter_estimate` ile birebir, testle korunuyor) |

CPU kısıtı (4C/6T, 7,6 GB, CUDA yok) mimariyi belirledi: 0,94 M parametre,
256 token bağlam, batch 16 → **0,74 s/adım**.

## 2. Veri yükleyici

Token dizileri 312–4491 arasında; model sabit bağlam ister.
[dataset.py](../ophelia/model/dataset.py) her dosyayı `context + 1` token'lık
pencerelere böler:

- uzun dosyalar: `stride = context // 2` (50% örtüşme) → 2.922 eğitim penceresi
- **kuyruk penceresi**: her dosya için tam bağlamlı bir pencere tam dosya
  sonunda biter. Bu olmadan `EOS` yalnızca 1-4 token'lık kırıntı pencerelerde
  görünüyor, model **eseri bitirmeyi hiç öğrenemiyordu** (ilk eğitimde üretim
  hep token bütçesiyle kesiliyordu).
- kısa dosyalar `PAD` ile doldurulur; dolgu hedefleri kayıpta yok sayılır
- pencere asla dosya sınırını aşmaz (testle korunuyor)

## 3. Eğitim

`AdamW (β=0.9/0.95, wd=0.05)`, 80 adım doğrusal warmup + kosinüs azalma,
gradyan kırpma 1.0, `torch.set_num_threads(6)`.

```bash
ophelia model train --steps 1200 --context 256 --d-model 128 --layers 3 \
                    --dropout 0.25 --weight-decay 0.05 --eval-interval 100
```

Çıktılar (`models/ophelia-v3/`):

```
best.pt            en düşük doğrulama kaybındaki ağırlıklar
last.pt            son adım (kesintiden sonra devam için)
best.json          adım, kayıp, parametre sayısı
train_log.json     tüm yapılandırma + adım geçmişi
vocab.json         üretimde zorunlu sözlük
dataset_meta.json  tokenizer ayarları (ızgara, sınırlar, velocity binleri)
```

Sözlük ve künye **eğitim başlamadan** yazılır: eğitim yarıda kesse bile
checkpoint tek başına kullanılabilir kalır.

### Deneyler — neden bu boyut?

İlk deneme (2,3 M parametre, dropout 0,1) küçük veri setini **ezberledi**:

| Model | Parametre | Dropout / wd | Adım | En iyi val | Son eğitim |
| --- | --- | --- | --- | --- | --- |
| v1 | 2.296.704 | 0,10 / 0,01 | 600 (kesildi) | 1,837 (adım 150) | 0,06 → **aşırı öğrenme** |
| v2 | 940.160 | 0,25 / 0,05 | 800 | 1,561 | 1,508 |
| **v3** | **940.160** | **0,25 / 0,05** | **1200** | **1,463** | **1,509** |

v1'de eğitim kaybı 0,06'ya inerken doğrulama kaybı 1,84 → 3,39'a çıktı: model
koralleri kopyalıyordu. Küçültme + güçlü düzenlileştirme farkı kapattı; v3'te
eğitim ve doğrulama kaybı neredeyse eşit (1,51 / 1,46 — eğitimde dropout açık
olduğu için eğitim kaybı biraz yüksek).

v3 doğrulama eğrisi: 3,33 (adım 100) → 1,79 → 1,68 → 1,65 → 1,61 → 1,57 →
1,54 → 1,51 → 1,49 → 1,48 → 1,47 → **1,46 (adım 1200)**. Eğri hâlâ iniyor:
model **az eğitilmiş**, süre 14,9 dk olduğu için uzatmak ucuz.

## 4. Üretim

```bash
ophelia model generate -m models/ophelia-v3/best.pt \
    --tokens 1200 --temperature 0.95 --top-k 40 --min-notes 120 --seed 3
```

Akış: `<bos>` → otoregresif örnekleme → `decode_notes` → `write_midi`.

### Dilbilgisi kısıtlı örnekleme

Serbest örnekleme, eğitimde hiç görülmemiş diziler üretir (`PITCH` olmadan
`DURATION` gibi) ve çözümleyici hata verir — yani "MIDI'ye çevrilemeyen beste".
[grammar.py](../ophelia/model/grammar.py) 4. aşamadaki kodlayıcının dilini bir
durum makinesi olarak tutar ve her adımda geçersiz token'ların olasılığını
`-inf` yapar:

```
BOS → TIME → [PROGRAM] [CHANNEL] → (VELOCITY PITCH DURATION)+ → EOS
```

- Model yine kendi dağılımından örnekler; sadece dilbilgisi dışına çıkamaz.
- `EOS`, `min_notes` (varsayılan 120) nota üretilmeden yasak: ne boş eser ne de
  birkaç notalık kırıntı çıkar. Bu eşik olmadan kuyruk pencereleri `EOS`'u fazla
  güçlendirdiği için parçalar 44 notada (5,5 sn) bitiyordu.
- Güvenlik ağı: kısıt kapatılırsa (`--free`) bozuk parçalar atlanır, üretim
  çökmez; hiç nota çıkmazsa açık hata verilir.

### Dinlenebilirlik

Üretilen dosya bağımsız bir okuyucuyla (`mido`) doğrulandı: **tip 1 MIDI,
480 PPQ, 100 BPM, 106 `note_on` / 106 `note_off` (dengeli), 51,5 sn** —
`data/midi/out/ophelia_demo.mid`; herhangi bir DAW veya ses kütüphanesi
doğrudan çalar.

## 5. Üretim sonuçları

`--tokens 1200 --min-notes 120`, sıcaklık 0,95, top-k 40:

| Tohum | Token | Nota | Süre | Ton |
| --- | --- | --- | --- | --- |
| 1 | 574 (EOS) | 113 | 21,6 sn | A# major |
| 2 | 430 (EOS) | 93 | 15,6 sn | G natural_minor |
| 3 | 409 (EOS) | 100 | 51,5 sn | C major |
| 4 | 411 (EOS) | 96 | 41,4 sn | F major |

Dördü de `EOS` ile kendiliğinden bitti (bütçe zorlamadı).

### Müzikal ölçütler — gerçek korallerle karşılaştırma

| Ölçüt | Gerçek koraller (4 dosya) | Üretilen (v3, 4 dosya) |
| --- | --- | --- |
| Perde aralığı | 41–77 | 41–86 |
| Medyan perde | 60–64 | 60–67 |
| Ortalama nota süresi | 367–769 tick | 387–766 tick |
| Nota sayısı | 199–429 | 93–113 |
| Süre | 28,8–52,8 sn | 15,6–51,5 sn |

Süre, perde ve nota süresi dağılımları gerçek korallere yakın; eksik olan
**uzunluk** — üretilen parçalar gerçek bir koralin yarısı kadar.

## 6. Bilinen sınırlar

- **Az eğitim**: 1.200 adım ≈ 6,6 epoch; doğrulama eğrisi hâlâ iniyor. Daha
  uzun eğitim (3.000+ adım, ~40 dk) doğrudan kazanç.
- **Veri seti tek stilli**: 4 sesli, tamamı 4/4 Bach koralleri. Model koral
  üretir; başka stil istemek için PDMX gerekli.
- **Uzun vadeli yapı yok**: 256 token bağlam (≈ 85 nota) — parça boyunca tema
  gelişimi kuramaz; cümle düzeyinde tutarlı, eser düzeyinde değil.
- **Tek kanal**: normalizasyon kanalları birleştiriyor; çok enstrümanlı düzenleme
  7. aşamada (`CHANNEL_*` token'ları hazır).
- **Ölçü/vuruş token'ı yok**: veri %100 4/4 olduğu için şimdilik sorun değil.
- **CPU**: 1.200 adım 15 dk; GPU'ya geçilirse aynı kod çalışır (cihaz seçimi
  henüz otomatik değil, `cpu` sabit).
