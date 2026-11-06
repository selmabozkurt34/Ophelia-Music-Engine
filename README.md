# Ophelia Music Engine

Doğal dilde verilen komutları anlayarak **özgün müzik besteleyen**, besteyi farklı
enstrümanlara dağıtan, düzenlenebilir hâle getiren ve profesyonel ses dosyalarına
dönüştüren bağımsız bir yapay zekâ müzik motoru.

> **Durum:** 1-7. aşamalar tamamlandı. Model beste üretiyor, doğal dille
yönlendirilebiliyor, enstrümanlara dağıtılıp stereo WAV'a basılabiliyor ve çıktı
> **nota olarak (MusicXML)** dışa aktarılabiliyor. Sırada 8. aşama (mixing /
mastering).

## Hızlı başlangıç

```powershell
# 1) Sanal ortamı etkinleştir
.\.venv\Scripts\Activate.ps1        # PowerShell
# source .venv/Scripts/activate     # Git Bash

# 2) Ortamı doğrula
python scripts/check_env.py

# 3) Testleri çalıştır
python -m pytest
```

Sıfırdan kurulum için: `scripts/setup_env.ps1` (PowerShell) — ayrıntılar
[docs/01_environment.md](docs/01_environment.md) içinde.

## Komut satırı

```bash
python -m ophelia env      # donanım + ortam raporu
python -m ophelia info     # proje özeti ve yapılandırma
python -m ophelia paths    # dizin yapısını oluştur ve listele
python -m ophelia theory   # gam + diyatonik akorlar + progresyon
python -m ophelia data     # veri seti araçları (sources/build/report/verify)
python -m ophelia config   # ham YAML yapılandırmasını yazdır
```

Müzik teorisi katmanının canlı örneği:

```bash
ophelia theory --key "C major" --progression "I-V-vi-IV"
ophelia theory -k "A minor" -p blues --seventh
```

```text
Gam          : C4 D4 E4 F4 G4 A4 B4
Diyatonik akorlar
  I     C     Cmaj7    tonic
  ii    Dm    Dm7      subdominant
  ...
Progresyon
     I  C      (tonic)
     V  G      (dominant)
    vi  Am     (tonic)
    IV  F      (subdominant)
```

`pip install -e .` sonrasında aynı komutlar `ophelia ...` olarak da çalışır.

## Donanım profili ve sonuçları

| Bileşen | Değer |
| --- | --- |
| CPU | Intel i3-1315U (4C / 6T) |
| RAM | 7,6 GB |
| GPU | Intel UHD — CUDA yok |
| Python | 3.11.9 (`winget`) |
| VS Code | 1.143.1 (`.vscode` ayarları hazır) |
| PyTorch | 2.15.0 **CPU** |

CUDA bulunmadığından 5. aşamadaki model mimarisi küçük tutulacak
(≤ 10 M parametre, 512–1024 token bağlam). Ayrıntılı gerekçe:
[docs/01_environment.md](docs/01_environment.md).

## Proje yapısı

```
ophelia/
├── core/     # yollar, YAML yapılandırma, günlükleme, ortam denetimi
├── theory/   # gam, akor, armoni, ritim, form, enstrümanlar (hazır)
├── data/     # lisans, temizleme, normalizasyon, bölütleme, tokenizer (hazır)
├── model/    # beste modeli, eğitim, dilbilgisi kısıtlı üretim (hazır)
├── nlp/      # doğal dil → parametre → model (hazır)
├── audio/    # enstrüman dağıtımı + sentez + stereo miks (hazır)
├── notation/ # MusicXML dışa aktarımı — düzenlenebilir nota (hazır)
├── ui/       # 9. aşama — kullanıcı arayüzü
└── cli/      # komut satırı arayüzü
```

Yol haritasının tamamı ve aşama durumları: [docs/ROADMAP.md](docs/ROADMAP.md).
Piyasa karşılaştırması ve konum analizi: [docs/piyasa_analizi.md](docs/piyasa_analizi.md).

## Yapılandırma

Tüm parametreler [configs/default.yaml](configs/default.yaml) içinde tutulur; kod
içine gömülü sabit yoktur. Deneyler için dosyayı kopyalayın:

```bash
cp configs/default.yaml configs/deney_01.yaml
python -m ophelia config --file configs/deney_01.yaml
```

## Veri seti

Tohum veri seti, lisans zinciri en net kaynaktan üretildi: **Bach koralleri**
(eserler kamu malı, kodlamalar BSD-3 lisanslı `music21` paketinden).

```bash
python scripts/fetch_corpus.py     # korpusu MIDI'ye çevir (data/raw)
ophelia data build                 # temizle → normalize → böl → raporla
ophelia data verify                # manifest ↔ disk bütünlüğü
```

| Ölçüt | Değer |
| --- | --- |
| Dosya | 427 (ham 430, 2 kalite, 1 tekrar elendi) |
| Süre | 4,56 saat |
| Nota | 116.713 |
| Polifoni | ort. 4,21 |
| Bölünme | train 340 · val 49 · test 38 |
| Token | 393.839 (sözlük 2452, ort. 922/dosya, en uzun 4.491) |
| Kayıpsız gidiş-dönüş | **427/427 (%100)** |

## Lisans politikası

Yalnızca **ticari kullanıma uygun** kaynaklar kullanılır. Kararlar
[ophelia/data/sources.py](ophelia/data/sources.py) içinde kodludur ve üretim hattı
onaysız bir kaynağı **reddeder** (örn. MAESTRO, MetaMIDI, Lakh).

```bash
ophelia data sources            # 11 kaynağın lisans kararı ve gerekçesi
ophelia data sources -s rejected
```

## Token temsili

MIDI, olay tabanlı bir token dizisine çevriliyor ve geri kuruluyor; dönüşüm
**kayıpsız** (427/427 dosya doğrulandı):

```bash
ophelia data tokenize --verify   # tokenize et + kayıpsızlığı doğrula
ophelia data roundtrip           # yalnızca doğrulama
```

Sözlük 2452 token: `TIME`, `DURATION`, `VELOCITY`, `PITCH`, `PROGRAM`, `CHANNEL`.
Akorlar tek zaman token'ı altında gruplanır. Ayrıntı: [docs/04_representation.md](docs/04_representation.md).

Her dosyanın kökeni ve sha256 özeti `data/manifests/` altında tutulur. Veri
dosyaları telif/boyut nedeniyle depoya eklenmez. Ayrıntı: [docs/03_dataset.md](docs/03_dataset.md).

## Beste modeli

CPU'ya uygun küçük bir decoder-only transformer (**940.160 parametre**, 256 token
bağlam) token veri setiyle eğitildi ve ürettiği token'ları dinlenebilir MIDI'ye
çeviriyor:

```bash
ophelia model train                # ~15 dk (CPU) → models/ophelia-*/best.pt
ophelia model generate --seed 3    # → data/midi/out/*.mid (DAW'da açılabilir)
ophelia model info                 # mimari, adım, val kaybı
```

| Ölçüt | Değer |
| --- | --- |
| En iyi doğrulama kaybı | **1,463** (1.200 adım, 14,9 dk) |
| Üretim | 93–113 nota · 15–52 sn · EOS ile kendiliğinden bitiş |
| Kısıt | Dilbilgisi kısıtlı örnekleme — çözümlenemeyen dizi üretilemez |
| Doğrulama | Üretilen `.mid` bağımsız `mido` ile okundu (480 PPQ, 100 BPM) |

Ayrıntı (deneyler, aşırı öğrenme, müzikal ölçütler): [docs/05_model.md](docs/05_model.md).

Model ağırlıkları ve üretilen MIDI'ler depoda tutulmaz (boyut); `train` ve
`generate` komutları bunları yeniden üretir.

## Doğal dil komutları

```bash
ophelia compose "hüzünlü bir la minör parça, yavaş, 30 saniye"
ophelia compose "neşeli do majör 120 bpm 25 saniye" --explain
ophelia compose "dramatik mi bemol minör 90 bpm, dört ses" --seed 7
```

Anlaşılan alanlar: **tonalite** (`la minör`, `am`, `Bb major`), **tempo**
(`120 bpm`, `yavaş`, `andante`), **süre** (`30 saniye`, `8 ölçü`, `uzun`),
**doku** (`solo`, `dört ses`, `dolu`) ve **ruh hâli** (`hüzünlü`, `neşeli`,
`sakin`). Açıkça yazılan değer her zaman ruh hâlinin varsayılanını yener;
`--explain` her kararın gerekçesini gösterir.

| Garanti | Ölçüm |
| --- | --- |
| İstenen tonalite | Perdelerin **%100'ü** istenen gamda (aktarım + diyatonik projeksiyon) |
| İstenen süre | Tam (kırpma sınırda; EOS cümle sonu sayılıp üretim sürüyor) |
| İstenen tempo | MIDI `set_tempo` olarak yazılı |

Ayrıntı: [docs/06_nlp.md](docs/06_nlp.md).

## Enstrüman ve ses

Tek kanallı beste seslere ayrılıp enstrümanlara dağıtılıyor ve stereo WAV'a
basılıyor — **hiçbir nota atılmaz**. Düzenleme ya perde bölgesine göre ya da
komutta verilen **rollere** göre (melodi → armoni → bas) yapılıyor.

```bash
ophelia compose "hüzünlü la minör, yavaş, 25 saniye, yaylılar ve kontrbas" --wav
ophelia compose "melodi, armoni ve bas ile la minör, 20 saniye" --wav
ophelia audio render outputs/midi/eser.mid -e violin,viola,cello,contrabass
ophelia audio render outputs/midi/eser.mid -r melodi,armoni,bas
ophelia audio instruments          # katalog: rol, GM program, ses aralığı
```

| Ölçüt | Değer |
| --- | --- |
| MIDI | tip 1 · ses başına iz · program change + CC10 pan |
| Nota korunumu | 165 → 165 (bölge ayırma kayıpsız) · rol modunda 70 → 24/24/22 |
| WAV | 44 100 Hz · stereo · 16-bit · tepe −1 dBFS |

Ayrıntı (ses ayırma, sentez zinciri, bulunan hatalar): [docs/07_audio.md](docs/07_audio.md)

## Notasyon: MusicXML çıktısı

MIDI dinlenebilir ama **düzenlenebilir nota** değil. Aynı eser MusicXML olarak da
yazılıyor; MuseScore, Dorico, Sibelius ve Finale bu biçimi doğrudan açıyor.

```bash
ophelia compose "hüzünlü la minör, yavaş, 25 saniye, yaylılar" --musicxml
ophelia notation export outputs/midi/eser.mid -o outputs/scores/eser.musicxml
ophelia notation export eser.mid --title "Deneme 1"
```

| Ölçüt | Değer |
| --- | --- |
| Biçim | MusicXML 4.0 · partwise · UTF-8 |
| Ölçüleme | 4/4 varsayılan; ölçü çizgisini aşan nota bağla bölünür, boşluklar susla dolar |
| Çokseslilik | eşzamanlı notalar `<chord/>`, üst üste binen sesler ayrı `<voice>` + `<backup>` |
| Anahtar imi | notalardan tahmin (Krumhansl-Kessler → `<fifths>`), sol/sa anahtarı ses ortalamasına göre |
| Sadakat | geri okunduğunda perde · başlangıç · süre **birebir** (bağlar çözülerek doğrulandı) |
| Sınır | nota başına velocity taşınamaz; ortalama velocity tek `<sound dynamics>` olarak yazılır |

Gerekçe ve notasyon pazarı: [docs/piyasa_analizi.md](docs/piyasa_analizi.md)..
