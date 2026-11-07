# 8. Aşama — Mixing / mastering (ses karakteri)

Dinleyici geri bildirimi netti: çıktı **"oyun müziği gibi"** geliyordu. Ölçüm bunu
doğruladı — sorun kompozisyonda değil, seste. Kör testte model üretimi gerçek
korallerden istatistiksel olarak ayrışmıyordu (tercih %33, GA 9,7–70,0), yani
notalar fena değildi; kötü olan tını ve mekândı.

## 1. Teşhis — "oyun müziği" nereden geliyordu

7. aşamanın zinciri bilinçli bir iskeletti (`render.py`): saf sinüs harmonikleri,
8 ms ataklı üstel sönüm, notalar arasında **boşluk**, mekân yok. Koral müzikte
bu dördü birlikte "müzik kutusu / chiptune" etkisi yapar:

| Belirti | 7. aşamadaki neden |
| --- | --- |
| Notalar birbirine bağlanmıyor | her nota tam süresinde bitiyor, 90 ms'lik zorla sönüm |
| Vurmalı, çınlayan tını | saf sinüs serisi + 8 ms atak + üstel sönüm (sustain yok) |
| Kuru, "kafanın içinde" | reverb yok |
| Tekdüze, mekanik | velocity sabit, vibrato yok, dinamik işleme yok |
| Konser salonu değil, hoparlör | pan dışında mekân bilgisi yok |

## 2. Ne değişti

### 2.1 Tını — `ophelia/audio/timbre.py`

- Harmonik seri 1/n^1,05 ile sönüyor (12 harmonik); çift harmonikler hafifçe
  zayıflatılmış → yaylı/koa spektrumu, org değil.
- **Chorus**: ±6 sent detune'lu ikinci katman; tek osilatörün "beep" karakterini kırar.
- **Vibrato** 5,1 Hz / 9 sent, 0,45 sn'de oturarak (insan yaylısı gibi, anında değil).
  Faz birikimli hesaplanır: frekans kayması tıkırtı üretmez.
- **ADSR**: atak → sönüm → **sustain platosu** → uzun bırakma. Sustain, üstel
  sönümün "çınlama" hissini kaldıran şey.
- Koro dokusunda çok düşük seviyeli nefes gürültüsü (alçak geçiren süzülmüş).

### 2.2 Legato — `render.py`

Aynı partide ardışık notalar arasındaki boşluk ≤ 350 ms ise nota, sonraki notanın
başlangıcının 90 ms ötesine kadar uzatılır. İki nota bir an birlikte tınlar: yay
değişimi / nefes alma. Ayrıca her notanın bırakma kuyruğu (0,38–0,85 sn) nota
süresinin dışına taşar, yani notalar arasında **sessizlik oluşmaz**.

### 2.3 Mekân — `ophelia/audio/reverb.py`

Sentetik salon impuls yanıtı: 2,2 sn üstel sönüm + 4 erken yansıma + alçak geçiren
gövde, iki kanal için ayrı gürültü (stereo genişlik buradan gelir). Kuru/ıslak
oranı 0,24. Tohum sabit → çıktı tekrarlanabilir.

### 2.4 Yayın zinciri — `ophelia/audio/mastering.py`

| Adım | Ne yapıyor |
| --- | --- |
| K-ağırlık | 38 Hz yüksek geçiren + 1,5 kHz RBJ rafı (+4 dB) — BS.1770 yaklaşımı |
| LUFS | Hedef **−14 LUFS**; gating'siz (belgelenmiş yaklaşıklık) |
| Kompresör | Bus, eşik −20 dB, 2:1, anlık atak + üstel bırakma (pompalanma yok) |
| True peak | 4× yeniden örnekleme ile örnekler arası tepe; tavan **−1 dBTP** |

```bash
ophelia audio render outputs/midi/eser.mid            # salon + −14 LUFS (varsayılan)
ophelia audio render eser.mid --reverb 0 --no-master  # 7. aşama davranışı
```

## 3. Ölçüm — aynı eser, iki zincir

Eser: `bwv248.42-4` (gerçek Bach korali, 15 sn, aynı düzenleme). "Eski" kolon
7. aşama zinciriyle üretilmiş A/B kaydı, "yeni" kolon bu aşamanın zinciri.

| Ölçüt | 7. aşama | 8. aşama | Yorum |
| --- | --- | --- | --- |
| LUFS | −13,86 | **−14,00** | yayın hedefi tutuyor |
| True peak | −2,95 dBTP | **−6,18 dBTP** | tavanda bol headroom |
| Sessizlik dip oranı (20 ms pencereler) | %0,8 | **%0,1** | notalar arası boşluk kapandı |
| Spektral merkez | 948 Hz | **1661 Hz** | harmonik zenginlik arttı (org → yaylı) |
| Kuyruk enerjisi (son %15 / ilk yarı) | +0,5 dB | **+1,1 dB** | salon duyuluyor |
| Stereo genişlik | 0,064 | **0,140** | mekân iki kanalda farklı |
| L/R korelasyonu | 0,92 | **0,55** | mono değil, gerçek stereo |

Kör test için üretilen 12 kaydın tamamı **20,00 sn · −14,00 LUFS** (true peak en
yüksek −3,21 dBTP) — yani "hangi kayıt daha yüksek" konfoundu kapandı: artık iki
taraf da eşit yükseklikte.

## 4. Bilinen sınırlar

1. **Tını hâlâ sentetik.** Örnek/SoundFont yok; yaylı *benzeri*, yaylı değil.
   Aşamanın kalan işi bu.
2. **LUFS gating'siz.** Sessiz bölümler ortalamayı aşağı çeker; hedef −14 LUFS
   pratikte birkaç dB oynayabilir.
3. **Tek salon.** Farklı mekân (oda/katedral) seçeneği yok; impuls yanıtı sabit.
4. **Dinamik yok.** Nota başına crescendo/accelerando hâlâ yok; velocity modelden
   geliyor ve tek düze.
5. **Kulakla doğrulanmadı.** Yukarıdaki tablo ölçümdür; "artık koral gibi
   duyuluyor" iddiası dinleyici testiyle sınanmalı (bkz. `docs/kor_ab_testi.md`).

## 5. İkinci tur — sorun tını değil, ritim çıktı

Salon ve yeni tını eklendikten sonra geri bildirim aynı kaldı: "hâlâ oyun
müziği gibi". Bu kez ölçtüm: gerçek Bach koralleri ile model çıktısı aynı
yöntemle çıkarıldı.

| Ölçüt | Gerçek koraller (8) | Model v1 (15) | Model v2 (6) |
| --- | --- | --- | --- |
| 16'lık nota oranı | **%3** | **%30** | **%0** |
| Güçlü vuruşta başlayan nota | **%78** | **%18** | %49 |
| Ortalama eşzamanlı ses | 2,52 | 2,25 | 2,63 |
| Nota/saniye | 8,39 | 5,42 | 4,90 |

Yani model, koral dokusunu 16'lık notalarla ve zayıf vuruşlarda örüyordu —
kulağa "riff/arpej" gibi gelen şey buydu; sentez bunu yalnızca parlatıyordu.

### Yapılan düzeltmeler

1. **Ritim politikası (üretimde).** `SampleConfig.min_duration_steps=2` ve
   `time_step_steps=2`: dilbilgisi artık 16'lık `DURATION` ve 8'lik ızgaraya
   oturmayan `TIME` token'larını üretemez (`ophelia/model/grammar.py`). Model
   kısıtlanmıyor, yalnızca dilbilgisi dışına çıkamıyor.
2. **Metrik düzeltme (üretim sonrası).** `ophelia/nlp/metrify.py`: başlangıçlar
   8'lik ızgaraya yuvarlanır, aynı perdede çakışanlar tekilleştirilir ve
   `limit_polyphony`'nin bıraktığı 1 ızgaralık kırıntılar düşürülür.
3. **İnsanlaştırma.** Notalar ±12 ms kayarak girer, güçlü vuruşta +%5 vurgu;
   tohum sabit olduğu için çıktı yine tekrarlanabilir.
4. **"Yaylılar" artık bölüm.** Komutta "yaylılar" dendiğinde tek bir pad
   yerine keman/viyola/çello/kontrbas ayrı ayrı çalıyor — tek tınıya düşen 111
   nota, "oyun müziği" hissinin diğer yarısıydı.

### Kalan fark (dürüst)

Güçlü vuruş oranı %49'da kaldı (gerçek: %78) ve model hâlâ gerçek korallerden
~%40 daha seyrek. Bu, **çözümleme** kısıtıyla kapanmaz; eğitim verisinin ritmik
çeşitliliği ve model kapasitesiyle ilgili. Sonraki adım: ritim ağırlıklı veri
(PDMX) ve daha büyük model.

## 6. Dosya dışa aktarımı

MP3/FLAC ve metadata bu aşamanın **son** adımı; henüz yok. Şimdilik çıktı 44,1 kHz
16-bit stereo WAV.
