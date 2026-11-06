# 2. Aşama — Müzik Teorisi Altyapısı

Modelin öğrenmesi ve kullanması gereken yapılar bu aşamada kodla tanımlandı.
Amaç, 4. aşamadaki veri temsilinin ve 5. aşamadaki modelin üzerine kurulacağı
**tek ve tutarlı bir teori sözlüğü** oluşturmaktı.

## Neden kendi katmanımız?

`music21` güçlü ama ağır (yüzlerce MB'lık bağımlılık zinciri, sembolik analizde
yavaş). Bu makinede 7,6 GB RAM ve CUDA'sız bir CPU var; veri hattında her MIDI
dosyası için akor/gam analizi yapılacağından sıcak yolda yalnızca saf Python
kullanılıyor. `music21` yalnızca **doğrulama** için opsiyonel referans olarak
kurulu kalıyor.

## Modüller

| Modül | Sorumluluk | Öne çıkan API |
| --- | --- | --- |
| [pitch.py](../ophelia/theory/pitch.py) | Nota ↔ MIDI ↔ frekans, oktav, enharmonik | `note_to_midi`, `midi_to_note`, `midi_to_freq`, `Pitch` |
| [chords.py](../ophelia/theory/chords.py) | Akor yapıları, sembol ayrıştırma, akor tanıma | `Chord`, `identify_chord`, `Chord.from_symbol` |
| [scales.py](../ophelia/theory/scales.py) | Gamlar/modlar, tonalite, diyatonik akorlar | `Key`, `Scale`, `estimate_key`, `roman_numeral` |
| [harmony.py](../ophelia/theory/harmony.py) | Progresyonlar, kadanslar, fonksiyonlar, ses yürütme | `progression`, `detect_cadence`, `voice_lead` |
| [rhythm.py](../ophelia/theory/rhythm.py) | Tempo, ölçü, alt bölünme, senkop, quantize | `TimeSignature`, `TimedNote`, `is_syncopated` |
| [form.py](../ophelia/theory/form.py) | Motif dönüşümleri, bölüm/form şablonları | `Motif`, `Form`, `build_form` |
| [instruments.py](../ophelia/theory/instruments.py) | Enstrüman rolleri, GM programları, aralıklar | `get`, `by_role`, `assign_channels` |

## Temel kararlar

### 1. Tek perde birimi: MIDI numarası
`C4 = 60 = 261.63 Hz`, `A4 = 69 = 440 Hz`. Metin gösterimi yalnızca giriş/çıkış
katmanında var. `B#3` gibi taşan yazımlar `C4` ile aynı sonucu verir (oktav
taşması korunur) — `#`/`b` işaretleri mod alınmadan toplanır.

### 2. Akorlar perde sınıfı kümesidir
`Chord = (kök, kalite, çevrim)`. Seslendirme istendiğinde `midi_notes()` çağrılır.
19 kalite tanımlı: majör/minör/dim/aug üçlüler, sus2/sus4, power5, 6'lı, maj7/m7/7,
m7b5, dim7, mM7, add9 ve 9'lular.

Akor tanıma **bas sesine öncelik verir**: `{C, E, G, A}` hem `C6` hem `Am7`'dir;
en pes nota kök ise `C6`, `A` ise `Am7` döner. Bu, gerçek MIDI verisinde en sık
karşılaşılan belirsizliği deterministik biçimde çözer.

### 3. Diyatonik akorlar gamdan türetilir
`Key.chord(degree)` gam üzerinde üçlü yığar ve sonucu `identify_chord` ile tanır.
Böylece akor bilgisi tek yerde kalır; C majör'de `I ii iii IV V vi vii°` ve
`Cmaj7 Dm7 Em7 Fmaj7 G7 Am7 Bm7b5` çıktıları testlerle sabitlendi.

Harmonik minörde `V` majör, `i` ise `mM7` olur — klasik armoni kuralı.

### 4. Tonalite tahmini Krumhansl-Kessler ile
`estimate_key`, perde sınıfı histogramını 12 majör + 12 minör profille
ilişkilendirir. 3-4. aşamada MIDI dosyalarından tonalite etiketi çıkarmak için
kullanılacak; sonuç her zaman majör ya da doğal minördür.

### 5. Zaman birimi: çeyrek vuruş
1.0 = çeyrek nota. `4/4` ölçü = 4.0, `6/8` = 3.0. Tempo yalnızca saniyeye
çevrimde devreye girer (`beats_to_seconds`). Quantize ve senkop analizi bu tek
birim üzerinden yapılır.

Senkop tanımı: **zayıf zamanda başlayıp bir sonraki vuruşu aşan nota**. Böylece
düz sekizlikler senkop sayılmaz, ama bağlanmış/uzatılmış zayıf vuruşlar sayılır.

### 6. Ses yürütme (voice leading)
`voice_lead`, ardışık akorlarda her sesi bir önceki akorun en yakın sesine
bağlar (ortak sesler korunur, hareket en aza iner) ve verilen aralıkta kalır.
7. aşamada kanal/enstrüman dağıtımının girdisi olacak.

## Kullanım

```bash
# C majör gamı, diyatonik akorlar ve bir progresyon
ophelia theory --key "C major" --progression "I-V-vi-IV"

# Hazır progresyon + yedili akorlar
ophelia theory -k "A minor" -p blues --seventh

# Sayısal dereceler
ophelia theory -k "Bb major" -p 1-6-4-5
```

Kod içinden:

```python
from ophelia.theory import Key, progression, voice_lead

key = Key.parse("D major")
chords = progression(key, "I-V-vi-IV")          # D, A, Bm, G
voicings = voice_lead(chords, low=55, high=79)  # ardışık hareket en az
```

## Doğrulama

`tests/test_theory_*.py` altında 128 test var; kapsam:

- notaların tam MIDI aralığında gidiş-dönüşü (0-127)
- 11 akor kalitesinin sembol ↔ yapı dönüşümü ve çevrimler
- C majör / A minör diyatonik akor tabloları (birebir)
- kadans tespiti, fonksiyon eşlemesi, ses yürütme aralık kontrolü
- ölçü/tempo dönüşümleri, quantize, senkop, swing
- motif dönüşümleri (invert, retrograde, augment) ve form şablonları
- enstrüman kaydı, GM programları, davulun 9. kanala sabitlenmesi

## Bilinen kısıtlar

- Kromatik/altere akorlar (`bVII`, `V/ii`, triton ikame) henüz yok — 4. aşamada
  token sözlüğü genişlerken eklenecek.
- `identify_chord` 2-5 sesten fazlasını ve yayılmış (spread) seslendirmeleri
  tanımaz; gerçek veride bas + üst yapı ayrı ayrı analiz edilecek.
- Tonalite tahmini perde sınıfı ağırlıklarına dayanır; ritmik vurgu ve süre
  ağırlığı henüz hesaba katılmıyor.
