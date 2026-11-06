# 4. Aşama — Müzikal Veri Temsili

Amaç: MIDI'yi modelin okuyabileceği bir **token dizisine** çevirmek ve bu
dönüşümün **kayıpsız** olduğunu kanıtlamak.

## 1. Neden olay tabanlı token'lar?

Üç aday değerlendirildi:

| Temsil | Artı | Eksi |
| --- | --- | --- |
| Piano roll (128×T matris) | Basit, CNN'e uygun | Uzun bağlamda T patlar; CPU'da bellek ağır |
| MIDI olayları (doğrudan) | Birebir | Değişken uzunlukta ikili veri, modellemesi zor |
| **Olay tabanlı token** | Kompakt, dil modeli gibi eğitilir | Sözlük tasarımı gerekir |

7,6 GB RAM / CUDA'sız CPU hedefi için üçüncü yol seçildi: model, bir dil modeli
gibi **sonraki token'ı tahmin** eder.

## 2. Sözlük

Sıra deterministiktir ([tokens.py](../ophelia/data/tokens.py)):

| Grup | Token | Adet |
| --- | --- | --- |
| Özel | `<pad>`, `<bos>`, `<eos>` | 3 |
| Zaman | `TIME_0 … TIME_1024` | 1025 |
| Süre | `DURATION_1 … DURATION_1024` | 1024 |
| Şiddet | `VELOCITY_0 … VELOCITY_127` | 128 |
| Perde | `PITCH_0 … PITCH_127` | 128 |
| Enstrüman | `PROGRAM_0 … PROGRAM_127` | 128 |
| Kanal | `CHANNEL_0 … CHANNEL_15` | 16 |
| **Toplam** | | **2452** |

Zaman ve süre birimi **ızgara**dır (varsayılan 120 tick = 480 PPQ'da 16'lık nota).
`TIME_k` "bir önceki olaydan k birim sonra" demektir; art arda gelen `TIME`
token'ları toplanır, böylece çok uzun boşluklar da temsil edilir.

## 3. Kodlama düzeni

Aynı başlangıç, program ve kanaldaki notalar **tek grup** oluşturur:

```
<bos>
TIME_0  PROGRAM_0  CHANNEL_0
  VELOCITY_90 PITCH_60 DURATION_1
  VELOCITY_90 PITCH_64 DURATION_1        ← akor: tek TIME altında
  VELOCITY_90 PITCH_67 DURATION_1
TIME_1
  VELOCITY_70 PITCH_62 DURATION_2
<eos>
```

Nota başına 3 token; akorlar tek zaman token'ı paylaşır.

Ölçülen dizi uzunlukları (427 dosya):

| | En kısa | Ortalama | En uzun | > 1024 token |
| --- | --- | --- | --- | --- |
| Tümü | 312 | **922** | **4.491** | **108 dosya (%25)** |
| train | 312 | 908 | 4.491 | |
| val | 448 | 955 | 3.682 | |
| test | 442 | 1.007 | 3.678 | |

**Sonuç:** tek parça hâlinde 1024 token'lık bağlama dosyaların yalnızca %75'i
sığar. 5. aşamada veri yükleyici ya 512–1024 token'lık **pencerelere bölecek**
(uzun eserlerde bağlam kaybı) ya da model bağlamı 2048'e çıkarılacak (CPU'da
maliyet ~2×). Karar 5. aşamada ölçümle verilecek.

## 4. Kayıpsızlık

Tanım: `decode(encode(notalar))` aynı notaları (başlangıç, süre, perde, şiddet,
kanal, program) verir. Sıra kanonikleştirilir, çünkü aynı anda farklı
programlardan gelen notaların dosyadaki sırası müzikal bilgi taşımaz.

**Ölçüm:** 427 dosyanın tamamı `encode → decode → karşılaştır` ile doğrulandı:

```bash
ophelia data roundtrip          # 427/427 kayıpsız (%100.0)
```

Izgara dışı veri sessizce bozulmaz: `strict=True` (varsayılan) `OffGridError`
fırlatır. Bu sayede 3. aşamadaki normalizer'ın **1 tick'lik hayalet notalar**
ürettiği hata yakalandı ve düzeltildi (ızgara altı süreler artık bir ızgara
birimine yuvarlanıyor).

## 5. Üretilen artefaktlar

`data/tokens/` (yeniden üretilebilir, `.gitignore`'da):

```
vocab.json           2452 token, sıralı liste
meta.json            yapılandırma + sayılar + dosya adları
train_tokens.npy     int32, bitişik   (308,749 token)
train_offsets.npy    int64, 341 adet  (dosya sınırları)
val_tokens.npy       46,824 token
test_tokens.npy      38,266 token
```

Eğitim tarafı tek bir düz diziden dilim alır:

```python
dataset = TokenDataset.load("data/tokens", tokenizer)
train = dataset.splits["train"]
sequence = train.sequence(0)        # int32 numpy dizisi
```

Bitişik dizi yaklaşımı, dosya başına ayrı `.npy` okumaktan çok daha hızlıdır ve
7,6 GB RAM'de tüm veri seti (1,6 MB) belleğe rahat sığar.

## 6. Komutlar

```bash
ophelia data tokenize                 # tokenize et
ophelia data tokenize --verify        # tokenize + kayıpsızlık doğrulaması
ophelia data roundtrip                # yalnızca doğrulama
ophelia data roundtrip --limit 50 --grid 120
```

## 7. Sınırlar

- **Ölçü/vuruş token'ı yok.** 4/4 dışındaki müzikte model vuruş yapısını yalnızca
  zaman aralıklarından çıkarır. Veri seti %100 4/4 olduğu için şimdilik sorun
  değil; PDMX eklendiğinde `BAR`/`BEAT` token'ları gerekecek.
- **Şiddet tam çözünürlüklü** (128 token), sözlüğün %5'i. `velocity_bins=16`
  ile 16 token'a indirilebilir (kayıplı) — model küçüldükçe değerlendirilecek.
- **Tek kanal varsayımı.** Normalizasyon kanalları 0'a birleştirdiği için akor
  ve kanal ayrımı şimdilik yok; çok enstrümanlı veri (7. aşama) geldiğinde
  `CHANNEL_*` token'ları devreye girecek.
- Sözlük 2452 token; gömme katmanı 2452×128 ≈ 314 bin parametre, 10 M hedefinin
  %3'ü — kabul edilebilir.
- **Uzun kuyruk:** 108 dosya 1024 token'ı aşıyor (en uzunu 4.491). Pencereleme
  stratejisi 5. aşamada netleşecek; şu an dizi başına tam eser tutuluyor.
