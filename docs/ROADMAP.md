# Ophelia Music Engine — Yol Haritası

İlk öncelik: **çalışan bir beste modeli** (5. aşama). Diğer tüm sistemler bu temel
üzerine kurulur. Müzik teorisi (2) ve veri hazırlama (3) paralel yürütülebilir.

| # | Aşama | Çıktı | Durum |
| --- | --- | --- | --- |
| 1 | Geliştirme ortamı | Python 3.11 + VS Code + venv + PyTorch (CPU) + dizin yapısı | ✅ Tamamlandı |
| 2 | Müzik teorisi altyapısı | `ophelia.theory` — gam, akor, armoni, ritim, form | ✅ Tamamlandı |
| 3 | Veri toplama ve hazırlama | Lisanslı MIDI veri seti + manifest kaydı | ✅ Tamamlandı |
| 4 | Müzikal veri temsili | Tokenizer / encoder + MIDI↔model köprüsü | ✅ Tamamlandı |
| 5 | Beste modeli | Eğitilmiş model, MIDI üretimi | ✅ Tamamlandı |
| 6 | Doğal dil komut sistemi | Komut → parametre → model | ✅ Tamamlandı |
| 7 | Enstrüman ve ses üretimi | Çok kanallı MIDI → ses | ✅ Tamamlandı |
| 8 | Mixing / mastering | Yayın kalitesinde WAV | ⬜ |
| 9 | Kullanıcı arayüzü | Uçtan uca çalışan uygulama | ⬜ |
| 10 | Ophelia.fm yayın hazırlığı | İlk 5 eser + ilk single | ⬜ |

## 1. Aşama — tamamlananlar

- [x] Donanım tespiti (CPU-only, 7,6 GB RAM) ve buna uygun PyTorch kurulumu
- [x] Python 3.11.9 ve VS Code 1.143.1 kurulumu, sanal ortam, sabitlenmiş bağımlılıklar
- [x] Proje dizin yapısı (veri, model, çıktı, belge, test)
- [x] Merkezî yol ve yapılandırma modülleri (`ophelia.core`)
- [x] Otomatik ortam denetimi (`scripts/check_env.py`, `python -m ophelia env`)
- [x] Kabul testleri (`tests/test_environment.py`)
- [x] VS Code yorumlayıcı/eklenti ayarları, Git deposu ve `.gitignore`

## 2. Aşama — tamamlananlar

- [x] `pitch.py` — nota ↔ MIDI ↔ frekans, oktav, enharmonik, `Pitch` değer nesnesi
- [x] `chords.py` — 19 akor kalitesi, sembol ayrıştırma/yazma, çevrimler, akor tanıma
- [x] `scales.py` — 13 gam/mod, `Key`, diyatonik akorlar, Roma rakamları, tonalite tahmini
- [x] `harmony.py` — 12 hazır progresyon, kadans tespiti, fonksiyonlar, ses yürütme
- [x] `rhythm.py` — ölçü/tempo, alt bölünme, quantize, senkop, swing
- [x] `form.py` — motif dönüşümleri (invert/retrograde/augment), 9 form şablonu
- [x] `instruments.py` — 23 enstrüman, roller, GM programları, kanal ataması
- [x] `ophelia theory` CLI komutu ve 128 test

Ayrıntılar: [02_music_theory.md](02_music_theory.md).

## 3. Aşama — tamamlananlar

- [x] 11 kaynağın lisans değerlendirmesi; kararlar `ophelia/data/sources.py` içinde kodlandı
- [x] Lisans kapısı: onaysız/NonCommercial kaynakla çalışmak **hata** veriyor
- [x] Manifest sistemi: `sources.json` (lisans + atıf) ve `files.jsonl` (sha256 + köken)
- [x] Tohum veri seti: music21 Bach koralleri → 430 MIDI (PD, temiz lisans zinciri)
- [x] Temizleme (ret sebep kodlarıyla), normalizasyon (PPQ 480, tek tempo, 16'lık ızgara)
- [x] Kararlı train/val/test bölütleme ve kalite raporu
- [x] **427 dosya · 4,56 saat · 116.713 nota · train 340 / val 49 / test 38**
- [x] `ophelia data sources|build|report|verify` komutları, 103 yeni test

Ayrıntılar: [03_dataset.md](03_dataset.md).

## 4. Aşama — tamamlananlar

- [x] Olay tabanlı sözlük: 2452 token (TIME, DURATION, VELOCITY, PITCH, PROGRAM, CHANNEL)
- [x] `MidiTokenizer`: akorları tek TIME altında gruplayan kodlama/çözme
- [x] `strict` mod: ızgara dışı veride hata → sessiz bozulma yok
- [x] Token veri seti: bitişik `int32` diziler + ofsetler (`data/tokens`, 1,6 MB)
- [x] **427/427 dosya kayıpsız gidiş-dönüş (%100)** — 393.839 token
- [x] `ophelia data tokenize|roundtrip` komutları, 33 test

Ayrıntılar: [04_representation.md](04_representation.md).

## 5. Aşama — tamamlananlar

- [x] Decoder-only transformer: 3 katman, d_model 128, 4 başlık, **940.160 parametre**
- [x] Pencere veri yükleyici: 50% örtüşme + her dosya için tam bağlamlı kuyruk
      penceresi (`EOS` öğrenilebilir olsun)
- [x] Eğitim döngüsü: AdamW, warmup + kosinüs, gradyan kırpma, checkpoint, val kaybı
- [x] Dilbilgisi kısıtlı örnekleme: model dilbilgisi dışı token üretemez
- [x] `decode_notes` → `write_midi`: üretim dinlenebilir MIDI (`mido` ile doğrulandı)
- [x] **Eğitim: 1.200 adım, en iyi val 1,463**, 14,9 dk (CPU)
- [x] `ophelia model train|generate|info` komutları, 58 yeni test (toplam 352)

Ayrıntılar: [05_model.md](05_model.md).

## 6. Aşama — tamamlananlar

- [x] `CompositionSpec`: komut ile model arasındaki tek sözleşme
- [x] Kural tabanlı, deterministik çözümleme (Türkçe + İngilizce, aksansız yazım dahil)
- [x] Tonalite tohumu: model istenen toniğin triadıyla başlıyor
- [x] EOS'u cümle sonu sayan devam üretimi (istenen süre dolana kadar, en çok 4 geçiş)
- [x] Aktarım + diyatonik projeksiyon + süreye kırpma + polifoni sınırı
- [x] `ophelia compose "..."` komutu, `--explain` ile karar gerekçeleri
- [x] **3 komutun 3'ünde de gam uyumu %100 ve süre tam**, 72 yeni test (toplam 424)

Ayrıntılar: [06_nlp.md](06_nlp.md).

## 7. Aşama — tamamlananlar

- [x] Ses ayırma: register bölgeleri (akor sırası seçeneğiyle) — **hiçbir nota atılmaz**
- [x] Enstrüman ataması: perde bölgesine göre sıralama, aralık düzeltmesi, kanal + pan
- [x] Çok kanallı MIDI (tip 1): program change + CC10 pan, ASCII-güvenli iz adları
- [x] Harmonik sentez: rol başına profil/zarf, velocity → genlik, eşit güçte pan
- [x] Stereo WAV (44,1 kHz · 16-bit), tepe −1 dBFS normalizasyonu
- [x] `ophelia audio render|instruments`, `compose --wav` ve komutta enstrüman adı
- [x] Rol tabanlı düzenleme: `melodi/armoni/bas` rolleri, role göre enstrüman seçimi (`pick_for_role`), `--roles`
- [x] **165 nota → 4 ses (56/30/41/38) · 4 program · 4 kanal · 30,4 sn stereo WAV**, 31 yeni test
- [x] Rol modu: 121 nota → 42/41/38, MIDI program 40/41/33, 20,4 sn stereo WAV
- [x] Perküsyon rolü açık hatayla reddedilir (davul kalıbı 8. aşamada)

Ayrıntılar: [07_audio.md](07_audio.md).

## Sonraki adım: 8. Aşama — Mixing / mastering

Ses artık üretiliyor ama "yayın kalitesi" değil: reverb yok, dinamik işleme yok,
sample tabanlı timbre yok.

1. Reverb / delay / kompresör: kanal ve master zinciri
2. SoundFont ya da örnek tabanlı enstrüman sesleri
3. Loudness hedefi (LUFS) ve gerçek tepe (true peak) ölçümü
4. WAV → MP3/FLAC dışa aktarım, metadata
