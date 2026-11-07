# Piyasa Analizi — Ophelia nerede, rakipler nerede

> **Tarih:** Kasım 2026
> **Kapsam:** ticari AI müzik platformları · açık kaynak ses ve sembolik modeller ·
> Türkiye müzik/AI pazarı · notasyon ve MIDI araçları
> **Yöntem:** birincil kaynak okuması — şirket fiyat/model sayfaları, model kartları,
> GitHub repoları, arXiv/IJCAI bildirileri, hukuk bürosu ve dernek (MESAM/MÜYAP/IFPI)
> açıklamaları. Kaynak numaraları metinde köşeli parantezle verilir, tam liste sonda.

Bu rapor iki soruyu yanıtlar: **(1)** projede ölçülmüş olarak neredeyiz, **(2)** bu
konum, gerçek rakiplerin bulunduğu yere göre ne anlama geliyor. İkinci sorunun kısa
yanıtı: *teknik olarak uçtan uca çalışan bir motor, ticari olarak henüz ürün değil* —
ama pazarın en kalabalık ve en riskli kısmında değil, savunulabilir bir nişte.

---

## 0. Yönetici özeti

1. **Hat çalışıyor, ölçülü ve uçtan uca:** komut → token → beste → enstrüman dağıtımı
   → stereo WAV. 479 test geçiyor, biçim kapısı temiz. Bu, pazarın çoğu "demo"sunun
   sahip olmadığı bir şey: **yeniden üretilebilirlik** (`train` + `generate` aynı
   sonucu verir) ve **kayıpsız veri sözleşmesi** (%100 token gidiş-dönüş).
2. **Ölçek farkı üç büyüklük mertebesi.** Veri 4,56 saat, model 0,94 M parametre.
   Karşılaştırma: MusicGen 20.000 saat lisanslı müzik [14], Magenta RT ~190.000 saat
   [19], ACE-Step 1.5 3,5–5 B parametre [22]. Ses modeli yarışına bu kaynakla
   girilmez; girilse bile kaybedilir.
3. **Pazarın merkezi "tam şarkı + vokal" ve hukuken sarsılıyor.** Suno ~$300M ARR ve
   2 M+ ücretli aboneye ulaştı [1] ama UMG/Sony davaları sürüyor, GEMA davasını
   kaybetti [6][7]; Udio indirmeleri tamamen kapattı [10]. Bizim ürettiğimiz şey
   (enstrümantal, sembolik, düzenlenebilir) bu fırtınanın dışında.
4. **Asıl rakibimiz Suno değil; MuseScore, Noteflight, Flat ve Dorico.** Kullanıcı
   "nota/MIDI üreten, düzenlenebilir, eğitimde kullanılabilir" bir araç arıyorsa
   bizimle bu ürünler arasında seçim yapıyor. Bu pazarda AI hâlâ **girdi tarafında**
   (OMR, audio-to-score, proofreading) para kazanıyor; üretken kompozisyon boş [34][36].
5. **Türkiye'de "Suno klonu" segmenti dolu ve finanse edilmemiş.** eita, Muzica,
   Velorafy, Hurrian AI gibi ürünler var; açıklanmış yatırım neredeyse yok [39][40].
   Buna karşılık MEB yaygın eğitimde "Müzik ve Gösteri Sanatları" 710 bin kursiyer
   [44] ve 105 güzel sanatlar lisesi [45] — **dağıtım kanalı B2C değil, eğitim.**
6. **Önerilen konum:** "lisans zinciri denetlenebilir, Türkçe komut alan, MIDI/nota
   çıktısı veren beste motoru" — ses kalitesi yarışına girmeden, eğitim ve notasyon
   ekosistemine entegre olan bir katman. Ayrıntı ve gerekçe §6.

---

## 1. Neredeyiz — ölçülmüş durum

### 1.1 Aşama tablosu

| # | Aşama | Durum | Ölçülmüş çıktı |
| --- | --- | --- | --- |
| 1 | Ortam | ✅ | Python 3.11.9, PyTorch 2.15 CPU, i3-1315U / 7,6 GB |
| 2 | Müzik teorisi | ✅ | 13 gam, 19 akor kalitesi, 12 progresyon, 23 enstrüman |
| 3 | Veri | ✅ | 427 dosya · 4,56 saat · 116.713 nota · train 340/val 49/test 38 |
| 4 | Temsil | ✅ | 2452 token sözlük · 393.839 token · **427/427 kayıpsız** |
| 5 | Beste modeli | ✅ | **940.160 parametre** · val **1,463** (1.200 adım, 14,9 dk CPU) |
| 6 | Doğal dil | ✅ | 3/3 komutta gam uyumu **%100**, süre **tam** |
| 7 | Enstrüman + ses | ✅ | 165 → 165 nota (kayıpsız) · 44,1 kHz stereo WAV · tepe −1 dBFS |
| 8 | Mixing/mastering | ⬜ | reverb, dinamik işleme, LUFS, sample timbre yok |
| 9 | Arayüz | ⬜ | — |
| 10 | Ophelia.fm | ⬜ | — |

### 1.2 Neyi gerçekten yapabiliyoruz

| Yetenek | Durum | Kanıt |
| --- | --- | --- |
| Sıfırdan beste (4 sesli koral dokusu) | ✅ | 93–113 nota, 15–52 sn, hepsi `EOS` ile kendiliğinden bitti |
| Türkçe doğal dil komutu | ✅ | `"hüzünlü la minör, yavaş, 30 saniye"` → spec; kural tabanlı, deterministik |
| Tonalite garantisi | ✅ | perdelerin %100'ü istenen gamda (aktarım + diyatonik projeksiyon) |
| Süre/tempo garantisi | ✅ | tam süre; MIDI `set_tempo` |
| Enstrüman/rol düzenlemesi | ✅ | melodi→keman, armoni→viyola, bas→bas gitar; nota kaybı yok |
| Dinlenebilir çıktı | ⚠️ | MIDI her DAW'da açılıyor; WAV harmonik sentez, timbre gerçekçi değil |
| Vokal | ❌ | yok |
| Stil çeşitliliği | ❌ | tek stil (Bach koralı); "caz/elektronik" isteği çözümlenir, üretim değişmez |
| Uzun vadeli form | ❌ | 256 token bağlam ≈ 85 nota; eser düzeyinde tema gelişimi yok |
| Nota (MusicXML) çıktısı | ✅ | `ophelia notation export` — MuseScore/Dorico'da açılabilen .musicxml (§8) |

**Dürüst cümle:** teknik prototip olarak "çalışıyor" iddiası savunulabilir; ürün
olarak "rekabet eder" iddiası savunulamaz. Aradaki fark model kalitesi değil,
**veri ve kanal** farkı.

---

## 2. Pazar haritası — ticari oyuncular

| Platform | Model (tarih) | Çıktı | Giriş fiyatı | Ölçek | Ticari kullanım | Hukuk |
| --- | --- | --- | --- | --- | --- | --- |
| **Suno** | v6 / v6-wild / v6-mini (9 Eyl 2026) | tam şarkı + vokal, 8 dk, stem, Studio DAW | $10/ay (yıllık $8) | 2 M+ ücretli abone, ~$300 M ARR, 100 M+ kullanıcı | ücretli planlarda, indirme kotalı | UMG+Sony derdest; WMG anlaştı; GEMA kaybı [4][5][6][7] |
| **Udio** | v1.5 Allegro (18 Mar 2025) — yenisi yok | tam şarkı + vokal, 48 kHz; **indirme kapalı** | $10/ay | $10 M seed, ~$500 M değerleme (2024) | fiilen yok | UMG/WMG anlaştı, Sony derdest [9][10][11] |
| **Stable Audio** | 3.0 (20 May 2026, açık ağırlıklı aile) | enstrümantal/multi-track, 6 dk, DAW eklentisi | Solo $12 → Studio $199 | Stability: toplam $232 M fon (Sony/UMG/WMG yatırımcı) | ücretli planlarda | dava yok, "tam lisanslı veri" [12][13] |
| **AIVA** | sürüm numarası yok; 250+ stil | **sadece enstrümantal + MIDI** | €11/ay (yıllık) | açıklanmadı | Pro'da telif **sahipliği** devri | dava yok [15] |
| **Mubert** | MusicGen v1 + Chat2Music v2 | enstrümantal/arka plan, 25 dk | Pro $39/ay | 100 M+ track (2023), ~100 K MAU; ~$2,3 M ciro (tahmin) | Pro/Business; DSP yayını **yasak** | dava yok [16] |
| **Soundraw** | v2 (10 Mar 2025) + 2026 güncellemeleri | enstrümantal, stem | $16,99/ay | 800 K+ yaratıcı (2024); ~$5 M fon | plana göre | dava yok; eğitim verisi şirket içi [17] |
| **Boomy** | sürümlü model adı yok | tam şarkı (basit vokal), stem yok | $14,99/ay | 21,98 M üretilen şarkı (site sayacı); $21,6 M fon | Creator/Pro'da tam | label davası yok; bot-stream skandalları [18] |
| **Loudly** | VEGA-1/2 (12 Mar 2026), VEGA-3, MANTA 1 | enstrümantal + vokal, stem, mastering | ~$10/ay | açıklanmadı (Berlin) | abonelik süresince; telif Loudly'de | dava yok [20] |

### 2.1 Bu tablodan çıkan üç yapısal gerçek

**(a) Pazar "tam şarkı"da yoğunlaşmış, kâr orada değil.** Sekiz üründen altısı tam
şarkı + vokal satıyor; fiyat bandı $10–40/ay ve neredeyse hepsi aynı üç özellikle
yarışıyor (vokal kalitesi, stem, indirme hakkı). Bu, bizim giremeyeceğimiz ve
girmememiz gereken bir yarış: 0,94 M parametreli, tek stilli bir modelle vokal
üretmek teknik olarak da lisans olarak da mümkün değil.

**(b) Hukuk, ürün yol haritasını belirliyor.** Suno'nun UMG/Sony dosyasında teorik
maruziyet ~$9 milyar olarak hesaplanıyor; GEMA kararı (31 Tem 2026) AB'de eğitim
verisi ve çıktı için emsal niteliğinde [6][7]. Udio'nun indirmeyi kapatması bir
"özellik kaybı" değil, **anlaşmanın bedeli** [10]. Bu, "temiz lisans zinciri"ni
etik bir tercih olmaktan çıkarıp **ticari bir varlığa** dönüştürüyor — ve bu varlık
bizde kodla korunuyor (`sources.py` lisans kapısı, 427/427 izlenebilir köken).

**(c) "Temiz veri" oyuncuları kurumsal segmentte avantaj kazanıyor.** Stability
Series B'sinde üç majör label'ın yatırımcı olması [13], Soundraw'ın "no scraped
catalogs" konumlandırması [17], Mubert'in DSP yasağı ile lisans güvenliği satması
[16] aynı eğilimin parçaları. Bu segmentte rekabet **model boyutuyla değil, lisans
dosyasının netliğiyle** yapılıyor — bizim en güçlü olduğumuz yer.

---

## 3. Açık kaynak ve araştırma tarafı

### 3.1 Ses (audio) modelleri

| Model | Parametre | Lisans (kod / ağırlık) | Donanım | Not |
| --- | --- | --- | --- | --- |
| MusicGen / AudioCraft | 300 M / 1,5 B / 3,3 B | MIT / **CC-BY-NC 4.0** | ≥16 GB VRAM | 20 K saat; FAD 4,88–5,48; ticari kapalı [14] |
| Magenta RT 2 | 2,4 B + 230 M | Apache-2.0 / CC-BY-4.0 | Apple Silicon, M3 Pro+ | gerçek zamanlı, MIDI kontrol, ~200 ms gecikme [19] |
| Stable Audio Open 1.0 | ~1,2 B | Stability Community License | 4090'da ~12,6 GiB | 47 sn, vokal yok, 486 K kayıt [12] |
| ACE-Step 1.5 | 2 B DiT + 0,6–4 B LM (XL ~5 B) | **MIT** | **<4 GB VRAM** | tam şarkı <2 sn (A100); açık kaynağın en verimlisi [22] |
| YuE2 | 3 B (3,63 B) | Apache-2.0 / **CC BY-NC 4.0** | 24 GB BF16 | **sembolik plan → ses**; açık modeller arasında en yüksek şarkı kalitesi [23] |
| DiffRhythm | 1,1 B DiT | Apache-2.0 (VAE kısıtlı) | 8 GB VRAM | 4:45 şarkı ~10 sn; FAD 2,11 [24] |
| MusicLM / Lyria | açıklanmadı | **kapalı / sadece API** | — | Lyria 3 model kartı: latent diffusion, Mart 2026 [25] |

### 3.2 Sembolik (MIDI/nota) modelleri — bizim gerçek ligimiz

| Model | Parametre | Lisans | Donanım | Çıktı |
| --- | --- | --- | --- | --- |
| **NotaGen / NotaGen-X** | 110 M / 244 M / **516 M** | **MIT** | 8 GB inference | **ABC nota** (MusicXML'e çevrilebilir); klasik dönem-besteci koşullu; CLaMP-DPO |
| Anticipatory Music Transformer | 128 M / **360 M** / 780 M | Apache-2.0 | tek GPU | **MIDI inpainting** — melodiye eşlik, köprü, varyasyon |
| Coconet (Bach Doodle) | açıklanmadı (~64 katman) | Apache-2.0 | tarayıcı (TF.js) | Bach koralı tamamlama |
| BachBot | ~1–2 M (3×256 LSTM) | MIT (ağırlık pratikte yok) | CPU | Bach koral harmonizasyonu; Turing testinde %51 |
| MuseGAN | açıklanmadı | MIT | tek GPU | 5 track piano-roll, 4 ölçü |
| Magenta (MusicVAE, GrooVAE) | çok küçük (13–100 MB ckpt) | Apache-2.0 | CPU | MIDI loop/interpolation/humanize; repo arşivli |

**Kritik kıyas:** Bizim modelimiz 0,94 M parametre. En yakın **aynı işi yapan**
(model sınıfı sembolik, amaç nota üretmek) açık model NotaGen: 110 M — **117 kat**
büyük; büyük sürümü 516 M — **549 kat** [26]. NotaGen 1,6 M parça ile ön-eğitilmiş,
8.948 klasik eserle koşullu eğitilmiş. Yani "bizim model küçük" tespiti doğru ama
asıl fark **veri**: 427 dosya vs 1,6 M parça.

Ses tarafındaki fark ise kategorik: MusicGen 20.000 saat [14], Magenta RT ~190.000
saat [19], bizim 4,56 saat — yani **~4.400× ile ~42.000×** arası. Bu, "daha çok
eğitim" ile kapanacak bir fark değil; farklı bir ürün tanımı gerektiriyor.

### 3.3 Değerlendirme: hangi metrik kime ait

- **Ses modelleri:** FAD (VGGish/CLAP), KLD, CLAP skoru, chroma benzerliği, PER
  (şarkı sözü anlaşılırlığı), MOS. FAD'in insan tercihiyle korelasyonu zayıf;
  FAD-CLAP-MA en iyi korelasyonu veriyor (ICASSP 2025) [28].
- **Sembolik modeller:** CLaMP-2 skoru, FMD, perplexity/NLL, MuseGAN deskriptörleri
  (EB/UPC/QN), insan A/B ve Turing testleri. LilyBench (2026) deskriptör ve embedding
  metriklerinin **sistematik olarak çeliştiğini** gösteriyor → tek skorla sıralama
  yanıltıcı [29].
- **Bizim için doğru metrik seti:** (1) gam uyumu, (2) süre hassasiyeti, (3) nota
  korunumu, (4) gerçek korallerle dağılım karşılaştırması (perde aralığı, medyan
  perde, nota süresi), (5) kör A/B (insan). İlk üçü zaten ölçülüyor ve %100; dördüncü
  kısmen (§1.1); beşinci **hiç yapılmadı** — bu raporun en büyük ölçüm boşluğu.

---

## 4. Türkiye pazarı

### 4.1 Yerli ürünler

| Ürün | Ne yapıyor | Finansman |
| --- | --- | --- |
| **eita AI** | Türkçe arayüzle metinden müzik/görsel/video; "ticari kullanım dahil" | açıklanmadı [39] |
| **Muzica** (İstanbul) | Türkçe + İngilizce şarkı; 33 tür, 8 vokal; web/iOS/Android | açıklanmadı [40] |
| **Velorafy** | B2B, Türk dilleri öncelikli, royalty-free, API "yakında" | açıklanmadı [41] |
| **Hurrian AI** | 19 tür; HL Music dağıtım entegrasyonu ("Spotify'a gönder") | topluluk projesi [42] |
| **AI Muzik Studio** | tek geliştirici, kredi paketi, abonelik yok | bootstrapped [43] |
| **Kahuna** (Londra, Türk kurucular) | streaming geliri verisiyle sanatçı finansmanı | **Yapay Zekâ Fabrikası seed** — açıklanmış tek müzik-AI yatırımı [46] |

Hepsi aynı ürünü satıyor: **prompt → şarkı**. Fiyat rekabeti global oyuncularla
(Suno $8–10) yapılıyor; hiçbirinin dağıtım avantajı, veri avantajı veya lisans
avantajı yok. Bu segmentte 5. oyuncu olmanın anlamı yok.

### 4.2 Pazar sayıları

| Gösterge | Değer | Kaynak |
| --- | --- | --- |
| Türkiye kayıtlı müzik geliri (2025) | **$144,3 M** (+%60,1), dünyada 24. | IFPI 2026 / MESAM-MÜYAP [47] |
| Streaming | $125,7 M (abonelik $86,2 M) | aynı |
| AI yükü | Deezer'a günde 75.000+ tamamen AI eser = yüklemelerin %44'ü | aynı |
| Sektör büyüklüğü (bağımsız) | €135,4 M, 994 işletme (2026) | IBISWorld [48] |
| Yaygın eğitim "Müzik ve Gösteri Sanatları" | 45.830 kurs / **710.245 kursiyer** (2023) | MEB [44] |
| Güzel sanatlar lisesi | 91 → **105** okul, 18.166 öğrenci (2023-24) | [45] |
| AI girişimi sayısı (Türkiye) | 315 (Haz 2023) → **504** (Eki 2026) | TRAI [49] |

**Yorum:** Türkiye'de "müzik üretim yazılımı pazarı" için güvenilir bir büyüklük
yok; olan rakamlar ya dernek cirosu ya eğitim kapasitesi. Ama 710 bin kursiyer +
105 lise + konservatuvar, **lisanslanabilir bir kullanıcı tabanı** demek. Flat for
Education ve Noteflight Learn tam olarak bu modelle (öğrenci başına yıllık lisans)
çalışıyor [34][35].

---

## 5. Notasyon ve MIDI araçları — asıl rakip kümesi

| Ürün | Fiyat | Kullanıcı | AI'ı nerede kullanıyor |
| --- | --- | --- | --- |
| **MuseScore Studio** | uygulama **ücretsiz/açık kaynak**; abonelikler $5,99/hafta–$19,99/ay | 12 M indirme, 1,5 M+ MAU | **NoteVision OMR** (~%95), **Audio-to-Score** (MP3→MSCZ) [34] |
| **Flat.io** | Free / Power (dinamik) / OMR kredisi | 5–6 M kullanıcı | PDF/foto → skor (kredi bazlı), gerçek zamanlı işbirlikte yazma [35] |
| **Noteflight** | Premium $7,95/ay veya $49/yıl | 7 M+ kullanıcı | üretken AI **yok**; SoundCheck performans değerlendirme [36] |
| **Soundslice** | Plus $5/ay, Teacher $20/ay | açıklanmıyor | otomatik senkron, otomatik stem, PDF tarama [37] |
| **Dorico 6** | Pro $579,99 (tek seferlik) | açıklanmıyor | **proofreading** (kural tabanlı hata denetimi) [38] |
| **Sibelius** | Artist $99/yıl, Ultimate $199/yıl | sektör standardı | PhotoScore / AudioScore (girdi otomasyonu) |
| **Magenta Studio** | ücretsiz, Apache-2.0 | niş | Continue/Generate/Interpolate/Groove/Drumify — **MIDI, yerelde** [50] |
| **Staccato** | $11,99–14,99/ay | niş | prompt/MIDI → 16 track MIDI; Dorico ve MuseScore ile uyumlu [51] |
| **LANDR Composer** | $8,25/ay'dan | niş | akor+bas+melodi MIDI üretimi, drag-and-drop [52] |

**Bu tablonun bizim için anlamı üç madde:**

1. **AI'ın bu pazarda para kazandığı yer "girdi":** fotoğraf→skor, ses→skor,
   hata denetimi, performans değerlendirme. Üretken kompozisyon neredeyse yok.
   MuseScore'un kullanım koşulları lisanslı içeriğin AI eğitiminde kullanımını
   açıkça yasaklıyor [34] — yani bu oyuncular üretken tarafa **gönüllü** girmiyor.
2. **Fiyatlar düşük ve süreklilik abonelikte.** $5–20/ay bandı, yıllık eğitim
   lisansları. Bizim maliyet yapımız (CPU'da 15 dk eğitim, sıfır API maliyeti)
   bu banda rahat oturuyor; Suno gibi GPU maliyeti yok.
3. **Format uyumu giriş bileti.** MusicXML/MIDI olmadan hiçbir notasyon aracına
   entegre olamayız. MIDI zaten vardı; MusicXML dışa aktarımı bu raporun
   yazılmasından sonra eklendi (`ophelia notation export`), yani giriş bileti
   artık elimizde.

---

## 6. Ophelia'nın konumu

### 6.1 Boyut bazında karşılaştırma

| Boyut | Ophelia | Suno v6 | Stable Audio 3.0 | ACE-Step 1.5 | NotaGen | MuseScore |
| --- | --- | --- | --- | --- | --- | --- |
| Çıktı | MIDI + sentez WAV | tam şarkı + vokal | enstrümantal ses | tam şarkı | **nota (ABC)** | notasyon |
| Model | 0,94 M | açıklanmadı | açıklanmadı | ~5 B | 110–516 M | — |
| Eğitim verisi | **4,56 saat** | açıklanmadı (milyonlarca kayıt iddiası) | tam lisanslı | açıklanmadı | 1,6 M parça | — |
| Lisans zinciri | **kodla denetlenen PD/CC0** | davalı | tam lisanslı | MIT | MIT | — |
| Türkçe komut | **var** | yok (prompt İngilizce) | yok | kısmi (50+ dil) | yok | yok |
| Deterministik kontrol | **var** (gam/süre/tempo %100) | yok | yok | kısmi | kısmi | — |
| Ses kalitesi | zayıf (harmonik sentez) | çok güçlü | güçlü | güçlü | — | — |
| Düzenlenebilirlik | **yüksek** (nota düzeyi) | orta (stem) | orta | orta | **yüksek** | **yüksek** |
| Donanım | **CPU, 7,6 GB** | bulut | bulut | <4 GB VRAM | 8 GB | — |
| Olgunluk | 7/10 aşama, 479 test | ticari | ticari | açık kaynak | araştırma | 12 M indirme |

### 6.2 Nerede gerideyiz (kapanmaz olanlar)

| Açık | Büyüklük | Kapanır mı? |
| --- | --- | --- |
| Veri ölçeği | 4,56 saat vs 20.000–190.000 saat (**4.400–42.000×**) | Kısmen: PDMX `no_license_conflict` 6.250 saat → 1.370× (tek adımda en büyük kazanç) |
| Parametre | 0,94 M vs 110 M–5 B (117–5.300×) | Evet, ama CPU'da 10–30 M sınıfına kadar; GPU olmadan değil |
| Ses gerçekçiliği | sample/neural vocoder yok | Evet: SoundFont + 8. aşama |
| Vokal | yok | Hayır, bu ürün tanımının dışında |
| Uzun form | 256 token ≈ 85 nota | Kısmen: bağlam 1024 + form planlayıcı |
| Stil çeşitliliği | tek stil | Veriye bağlı |

### 6.3 Nerede öndeyiz (savunulabilir)

1. **Lisans zinciri kodla denetleniyor.** 11 kaynağın kararı `sources.py` içinde;
   onaysız kaynak boru hattını **hata ile** durduruyor. Suno'nun $9 milyar teorik
   maruziyeti [5] ve Udio'nun kapanan indirmeleri [10] karşısında bu, satılabilir
   bir özellik.
2. **Türkçe doğal dil + deterministik garanti.** "La minör, 30 saniye, dört ses"
   dediğinizde gam uyumu %100 ve süre tam. Hiçbir ticari platform bunu taahhüt
   etmiyor; onlar örnekleme yapıyor.
3. **Nota düzeyinde düzenlenebilirlik.** Çıktı MIDI; DAW'da açılır, nota nota
   değiştirilir, yeniden orkestralanır. Ses modeli çıktısı bu esnekliği vermez.
4. **Sıfır marjinal maliyet.** Eğitim CPU'da 15 dk, üretim CPU'da; API faturası yok.
   $5–20/ay bandında sürdürülebilir.
5. **Ölçüm kültürü.** 479 test, kayıpsız gidiş-dönüş kanıtı, "bulunan üç gerçek
   hata" gibi belgelenmiş hata kayıtları. Yatırımcı/okul teknik incelemesinde bu
   nadir bir sinyal.

### 6.4 SWOT

| | |
| --- | --- |
| **Güçlü** | temiz lisans · Türkçe arayüz · deterministik kontrol · düzenlenebilir çıktı · CPU'da çalışma · test disiplini |
| **Zayıf** | veri 4,56 saat · 0,94 M parametre · ses timbre zayıf · vokal yok · tek stil · kör A/B testi yok |
| **Fırsat** | lisans krizi sonrası "temiz" talebi · eğitim kanalı (710 bin kursiyer) · notasyonda üretken AI boşluğu · Türk makamı · DAW/notasyon eklentisi |
| **Tehdit** | MuseScore/Sibelius üretken AI'ı kendi ekler · Suno lisanslı modelle hukuku çözer · açık modeller (ACE-Step MIT) hızla ucuzlar · tek kişilik ekip kapasitesi |

---

## 7. Strateji — hangi oyunu oynamalı

### 7.1 Oynanmayacak oyun: "Türkiye'nin Suno'su"

Gerekçe sayısal: vokal için ses modeli gerekir; ses modeli için 20.000+ saat veri
ve GPU gerekir; $8/ay fiyat bandında geri dönüşü yok. Ayrıca bu segmentte hukuk
riski devralınır. Bu yol, mevcut varlıkların hiçbirini kullanmıyor.

### 7.2 Oynanacak oyun: "denetlenebilir beste katmanı"

**Ürün cümlesi:** *Türkçe komutla, istediğin tonda ve sürede, nota düzeyinde
düzenlenebilir beste üreten; hangi veriyle eğitildiği belgelenmiş motor.*

Bu cümlenin üç ayağı var ve üçü de mevcut kodun üstüne oturuyor:

1. **Notasyon entegrasyonu** — MusicXML dışa aktarım (**yapıldı:**
   `ophelia notation export` / `compose --musicxml`) + MuseScore/Dorico'ya
   aktarılabilir çıktı. Rakip değil, eklenti olarak girilir; dağıtım hazır.
2. **Eğitim** — Türkçe arayüz, öğretmenin "30 saniyelik la minör 4 sesli örnek
   üret" diyebildiği, çıktının nota olarak açılabildiği bir araç. B2B/G2B lisans
   (öğrenci başına yıllık), B2C abonelik değil.
3. **Türk makamı** — Batı modeliyle rekabet etmek yerine, Türk müziği kuramı
   (makam, usul, perde) destekleyen tek motor olmak. Bu, global oyuncuların
   yapısal olarak giremeyeceği bir niş ve elimizdeki `theory` katmanı genişletilebilir.

### 7.3 Yol haritasına yansıması

| Öncelik | Adım | Gerekçe | Ölçüt |
| --- | --- | --- | --- |
| **P0** | ~~MusicXML dışa aktarım~~ **(yapıldı)** | notasyon pazarına giriş bileti; en ucuz kaldıraç | MuseScore'da açılıp çalınabilen .musicxml |
| **P0** | ~~Kör A/B altyapısı~~ **(yapıldı)** · 20+ dinleyici toplanıyor | hiç ölçmediğimiz tek kritik boyut | tercih oranı + güven aralığı |
| **P1** | 8. aşama: SoundFont + LUFS + reverb/kompresör | "dinlenebilir"den "sunulabilir"e | −14 LUFS, true peak ≤ −1 dBTP |
| **P1** | PDMX `no_license_conflict` (6.250 saat) | veri farkını 1.370×'e indirir; lisans temiz kalır | yeni stil çeşitliliği + val kaybı |
| **P2** | Bağlam 1024 + form planlayıcı | eser düzeyinde tutarlılık | 60+ sn yapısal tekrar tespiti |
| **P2** | Türk makamı kuram katmanı | savunulabilir niş | makam içi perde doğruluğu %100 |
| **P3** | 9. aşama arayüz (yerel web) | eğitim satışı için gerekli | uçtan uca demo |
| **P3** | 10. aşama Ophelia.fm | vitrin, gelir değil | 5 eser + 1 single |

**Yapılmaması gerekenler:** vokal modeli, ses modeli yarışı, "Suno klonu" Türkçe
arayüz, agresif B2C reklam. Bunlar mevcut varlıkları tüketir, fark yaratmaz.

---

## 8. Rakip iddialarını doğrulama planı

Pazardaki her sayı şirket beyanı veya üçüncü taraf tahmini. Kendi iddialarımızı
da aynı disiplinle ölçmeliyiz:

| İddia | Nasıl ölçülür | Sıklık |
| --- | --- | --- |
| "İstenen gamda" | perdelerin gam içi oranı (şu an %100) | her sürümde |
| "İstenen sürede" | MIDI süre farkı (şu an 0,0 sn) | her sürümde |
| "Nota kaybı yok" | giriş/çıkış nota sayısı | her sürümde |
| "Korale benziyor" | perde aralığı, medyan perde, nota süresi dağılımı | sürüm başına |
| "Dinleyici ayırt edemiyor" | kör A/B, ≥20 katılımcı | **ilk kez yapılacak** |
| "Hızlı" | CPU'da adım/sn, üretim sn/nota | sürüm başına |

---

## 9. Kaynaklar

**Ticari platformlar**
1. Music Business Worldwide — Suno "far beyond" 2 M abone / $300 M (1 Eki 2026):
   https://www.musicbusinessworldwide.com/suno-is-far-beyond-2m-subscribers-and-300m-in-revenue-says-mikey-shulman-who-argues-its-good-that-ai-music-has-no-firm-rules-of-the-road-yet/
2. TechCrunch — Suno $250 M Seri C, $2,45 B değerleme (19 Kas 2025):
   https://techcrunch.com/2025/11/19/legally-embattled-ai-music-startup-suno-raises-at-2-45b-valuation-on-200m-revenue/
3. Suno fiyatlandırma: https://suno.com/pricing
4. AI Lawsuits Tracker — UMG v. Suno: https://ailawsuittracker.com/cases/umg-v-suno
5. The Next Web — UMG+Sony ikinci dava, 60.202 kayıt:
   https://thenextweb.com/news/universal-sony-second-lawsuit-suno-60202-recordings
6. Simmons & Simmons — GEMA v. Suno kararı (31 Tem 2026):
   https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright
7. SKW Schwarz — GEMA/Suno analizi: https://www.skwschwarz.de/en/news/KI-Flash-GEMA-Suno
8. Reuters — WMG–Suno anlaşması (25 Kas 2025):
   https://www.reuters.com/legal/litigation/warner-music-group-settles-copyright-case-with-suno-licensed-ai-music-2025-11-25/
9. Reuters — UMG–Udio anlaşması (30 Eki 2025):
   https://www.reuters.com/business/media-telecom/universal-music-settles-copyright-dispute-with-ai-firm-udio-2025-10-30
10. Undetectr — Udio incelemesi (indirmeler kapalı, fiyatlar):
    https://undetectr.com/blog/udio-review-2026
11. Udio changelog: https://help.udio.com/en/articles/10748731-changelog-what-s-new-with-udio
12. Stable Audio Open model card: https://huggingface.co/stabilityai/stable-audio-open-1.0
13. Stability AI haber dizini (Series B $232 M, Stable Audio 3.0):
    https://stability.ai/news-updates
14. MusicGen model card: https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md
15. AIVA fiyatlandırma: https://www.aiva.ai/pricing
16. Mubert fiyatlandırma: https://mubert.com/render/pricing · 100 M track:
    https://musically.com/2023/07/12/ai-music-startup-mubert-reaches-100m-tracks-milestone
17. Soundraw fiyatlandırma/lisans: https://soundraw.io/pricing · https://soundraw.io/license
18. Boomy fiyatlandırma (CostBench, 28 Eki 2026):
    https://costbench.com/software/ai-music-generators/boomy/
19. Magenta RealTime 2 model kartı:
    https://github.com/magenta/magenta-realtime/blob/main/MODEL.md
20. Loudly lisans sözleşmesi: https://www.loudly.com/license-agreement
21. YuE2 repo: https://github.com/multimodal-art-projection/YuE
22. ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
23. YuE v1 README: https://github.com/multimodal-art-projection/YuE/blob/YuE-v1/README.md
24. DiffRhythm: https://github.com/ASLP-lab/DiffRhythm
25. Lyria 3 model kartı: https://deepmind.google/models/model-cards/lyria-3
26. NotaGen: https://github.com/ElectricAlexis/NotaGen · https://arxiv.org/abs/2502.18008
27. Anticipatory Music Transformer:
    https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html
28. FAD vs insan tercihi (ICASSP 2025): https://arxiv.org/abs/2506.19085
29. LilyBench (2026): https://arxiv.org/abs/2606.08722
30. Coconet: https://magenta.tensorflow.org/coconet
31. BachBot (ISMIR 2017): https://archives.ismir.net/ismir2017/paper/000156.pdf
32. MuseGAN (AAAI 2018): https://ojs.aaai.org/index.php/AAAI/article/view/11312
33. Magenta MusicVAE: https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/README.md
34. MuseScore Studio: https://mu.se/musescore-studio · fiyatlar:
    https://musescore.com/pro/landing/pricing
35. Flat.io: https://flat.io/pricing
36. Noteflight: https://www.noteflight.com/learn
37. Soundslice: https://www.soundslice.com/plans/
38. Dorico 6: https://blog.dorico.com/2025/04/dorico-6-released/ · Sibelius:
    https://www.avid.com/sibelius
39. eita AI: https://eita.tr/
40. Muzica: https://www.crunchbase.com/organization/muzica
41. Velorafy: https://www.velorafy.com/
42. Hurrian AI: https://hurrianai.com/blog
43. AI Muzik Studio: https://aimuzikstudio.com/en/about
44. MEB Hayat Boyu Öğrenme izleme raporu 2023:
    https://hbogm.meb.gov.tr/meb_iys_dosyalar/2024_05/14104707_izlemevedegerlendirmeraporu2023.pdf
45. Güzel sanatlar liseleri (dergipark): https://dergipark.org.tr/en/download/article-file/4591049
46. Dealroom — Kahuna seed: https://app.dealroom.co/news/feed/ai-fabrikas-backs-turkish-founders-london-music-fintech-startup-kahuna-in-seed-round
47. Independent Türkçe — IFPI 2026 / MESAM-MÜYAP verileri:
    https://www.indyturk.com/node/776353
48. IBISWorld Türkiye ses kaydı sektörü:
    https://www.ibisworld.com/turkey/industry/sound-recording-music-publishing/200261/
49. TRAI girişim haritası (2023): https://turkiye.ai/trai-yapay-zeka-girisimleri-haritasinin-haziran-2023-sonuclari-aciklandi
50. Magenta Studio: https://magenta.withgoogle.com/studio
51. Staccato: https://staccato.ai/pricing
52. LANDR Composer: https://www.landr.com/plugins/landr-composer

**Pazar büyüklüğü tahminleri (ihtiyatla):** notasyon yazılımı 2024'te $310,5 M
(Growth Market Reports) ile $1,5 B (DataHorizzon) arasında; müzik üretim yazılımı
$489 M (QYResearch) ile $8,44 B (Reanin) arasında — **17 kata kadar fark**. Bu
sayılar mutlak değer olarak değil, yalnızca yön (yıllık %5–10 büyüme) için
kullanılmalı.

---

## 10. Doğrulanamayanlar ve ihtiyat notları

1. **Suno/Udio güncel abone ve ciro rakamları** en son Şubat 2026'da şirket
   tarafından açıklandı; Ekim 2026'daki $5,4 B değerleme bilgisi rakip label'ların
   mahkeme dosyasından, Suno'nun kendi beyanı değil.
2. **AIVA, Mubert, Soundraw, Loudly, Boomy için bağımsız doğrulanmış ciro/kullanıcı
   verisi yok**; verilenler GetLatka/Prospeo gibi tahmin modelleri.
3. **Loudly ve Boomy resmi fiyat sayfaları JS ile render ediliyor**; rakamlar
   bağımsız doğrulama sayfalarından.
4. **Türkiye'ye özgü müzik teknolojisi yazılımı pazar büyüklüğü hiçbir kaynakta
   bulunamadı**; MEB ve güzel sanatlar lisesi rakamları PDF/JS panel kaynaklı,
   birincil sayfadan tam metin olarak doğrulanamadı.
5. **eita, Muzica, Velorafy, Hurrian AI için yatırım tutarları açıklanmamış.**
6. **Kör A/B testi**: altyapı kuruldu (§7.3 P0, `docs/kor_ab_testi.md`) ama
   **dinleyici verisi henüz yetersiz** — ilk oturum 6 deneme × 1 dinleyici.
   Rapordaki "dinleyici ayırt edemiyor" iddiası hâlâ doğrulanmış değil.

---

*Bu belge bir karar aracıdır, pazar vaadi değil. Sayılar kaynaklı; yorumlar
tartışmaya açık. Yeni bir kaynak okunduğunda §2 ve §3 tabloları güncellenmeli,
§7.3 öncelikleri yeniden gözden geçirilmelidir.*
