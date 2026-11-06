# 7. Aşama — Enstrüman ve Ses Üretimi

Model tek kanallı beste üretiyor. Bu aşama onu **duyulabilir bir düzeneğe**
dağıtıyor: seslere ayırma → enstrüman ataması → çok kanallı MIDI → stereo WAV.

```bash
ophelia compose "hüzünlü la minör, yavaş, 25 saniye, yaylılar ve kontrbas" --wav
ophelia compose "melodi, armoni ve bas ile la minör, 20 saniye" --wav   # rol tabanlı
ophelia audio render outputs/midi/eser.mid --ensemble violin,viola,cello,contrabass
ophelia audio render outputs/midi/eser.mid --roles melodi,armoni,bas
ophelia audio instruments            # katalog: kimlik, rol, GM program, aralık
```

```
tek kanallı notalar
   │  register bölgeleri (çeyreklik)
   ▼
sesler: keman · viyola · çello · kontrbas      (hiçbir nota atılmaz)
   │  aralık düzeltmesi, kanal + pan ataması
   ▼
çok kanallı MIDI (tip 1, program + CC10 pan)
   │  harmonik sentez, rol başına zarf, eşit güçte pan
   ▼
stereo WAV (44,1 kHz, 16-bit)
```

## 1. Seslere ayırma — neden bölge, neden akor sırası değil

İlk deneme "akor sırası" (SATB) yöntemiydi: aynı anda başlayan notalar tizden
pese sırayla seslere dağıtılır. Koral akorlarında doğru, ama modelin dokusu
çoğu zaman 1-2 eşzamanlı notayla ilerliyor — sonuç: **122 notanın 64'ü sadece
kemana** düşüyor, kontrbasa 9 nota kalıyordu.

Bölge yöntemi perdeleri çeyreklik dilimlere ayırır; her ses kendi bölgesinde
kalır. 165 notalık bir parçada dağılım **56 / 30 / 41 / 38** ve toplam yine 165.
`split_into_voices(..., mode="chord")` eski davranışı korur (test edilmiş).

## 2. Enstrüman ataması

Topluluk **perde bölgesine göre sıralanır**: komutta "kontrbas ve yaylılar"
dense de melodi tiz enstrümana gider. Ardından her nota enstrümanın çalınabilir
aralığına oktavlarla taşınır, taşınamıyorsa kırpılır.

Pan, sahneye yayılır: melodi hafif sağ (+0,45), bas hafif sol (−0,45).
Davul isteği desteklenir ama **9. kanal melodi çalmaz** (MIDI'de perküsyon
kanalıdır); kanal ataması bu kurala uyar.

## 3. Rol tabanlı düzenleme

Perde bölgesi yöntemi dengeli ama sağır: "melodi, armoni ve bas" dense de sesleri
rastgele üç banda böler, melodiye bas gitarı verebilir. Rol verildiğinde her rol
bir ses alır ve sıra korunur (melodi en tiz … bas en pes):

```
roller: melodi, armoni, bas  →  4 sesli doku 3 banda ayrılır
   │  pick_for_role: aynı roldeki aday → aralığı uyan aday → varsayılan
   ▼
keman (melody) · viyola (harmony) · bas gitar (bass)
```

Enstrüman seçimi üç kademelidir: (1) adaylar içinde **aynı roldeki** ve henüz
kullanılmamış enstrüman, (2) kalan adaylardan **orta noktası rolün doğal
aralığına düşen** enstrüman (çello 56 → kontrbas aralığında, bas gitar 44 →
keman aralığında değil), (3) hiçbiri uymuyorsa `ROLE_DEFAULTS`. Böylece
kullanıcının açıkça istediği enstrüman korunur, ama yanlış role zorlanmaz:
`"melodi ve bas"` + `bass` isteği keman + bas gitar verir.

Parçanın **müzikal rolü** enstrümanın etiketinden ayrı tutulur
(`Part.role` / `part_role`): çello bas çalıyorsa zarf ve harmonik profili de bas
olur. Ölçüm (`--seed 4`, A minör, 68 BPM, 20 sn, 121 nota):

| Ölçüt | Değer |
| --- | --- |
| Dağılım | keman 42 · viyola 41 · bas gitar 38 — toplam **121** |
| MIDI | tip 1 · 4 iz (meta + 3 ses) · program 40/41/33 · kanal 0-2 |
| WAV | 44 100 Hz · stereo · 16-bit · 20,4 sn · tepe 0,71 |
| `audio render --roles` | 70 nota → 24 / 24 / 22, kayıp yok |

Perküsyon rolü **reddedilir** (`"davul rolü henüz üretilmiyor"`): model davul
kalıbı üretmediği için melodic perdeleri davul kanalına yazmak sessizce çöp
üretirdi.

## 4. Sentez

Ses kütüphanesi değil, ölçülebilir bir iskelet: rol başına harmonik profili
(melodi 5 harmonik, bas 3, doku 4), rol başına zarf (yaylılarda 60 ms atak,
piyanoda 8 ms), velocity → genlik (`v^1.4`), eşit güçte pan, tepe
normalizasyonu (−1 dBFS).

Doğrulama (`mido` + `scipy.io.wavfile`):

| Ölçüt | Değer |
| --- | --- |
| MIDI | tip 1 · 5 iz (meta + 4 ses) · program 40/41/42/43 · kanal 0-3 |
| Nota korunumu | 165 → 165 · dağılım 56 / 30 / 41 / 38 |
| WAV | 44 100 Hz · stereo · 16-bit · 30,4 sn |
| Seviye | tepe 0,71 · RMS 0,097 · stereo genişlik 0,044 |

## 5. Yol boyunca bulunan üç gerçek hata

**1. Türkçe adlar MIDI'yi patlatıyordu.** İz adı olarak `instrument.name`
("Çello") yazılıyordu; MIDI iz adları latin-1 kodlanır ve `ı`/`Ç`
`UnicodeEncodeError` veriyordu. Artık ASCII kimlik yazılıyor (`cello`), görünen
ad raporlarda kalıyor.

**2. Melodi kontrbasa gidiyordu.** Bölge indeksi tersti: en pes notalar ilk
enstrümana (keman) gidiyordu. Düzeltildi, sonra topluluk da perde bölgesine
göre sıralandı.

**3. Notaların yarısı kayboluyordu.** Polifoni sınırı, sınırı aşınca notayı
**atıyordu** (185 → 100) ve "hüzünlü" sözcüğü ruh hâli tablosundan polifoni 4
getirdiği için bu kayıp normal akışta da oluyordu. Düzeltmeler: (a) aynı perde
çakıştığında önceki nota yenisinin başlangıcına kadar **kısaltılıyor** (MIDI'nin
"note steal" davranışı) — nota atılmıyor; (b) ruh hâli artık dokuya karışmıyor,
varsayılan sınır 6.

Ölçüm (30 sn'lik la minör parça, 226 kırpılmış nota): sınırsız **208** (yalnızca
18 birebir yinelenen nota düşer), sınır 8 → 189, sınır 6 → **165** (varsayılan),
sınır 4 → 124. Kayıp artık yalnızca kullanıcının istediği doku sınırından geliyor.

## 6. Bilinen sınırlar

- **Ses bir ses kütüphanesi değil**: sample yok, sadece harmonik sentez. Timbre
  gerçekçi değil; 8. aşamada SoundFont/örnek tabanlı sentez ve efektler gelecek.
- **Reverb/kompresör yok**: miks kuru; sadece seviye, pan ve tepe normalizasyonu.
- **Davul üretilmiyor**: model perküsyon üretmiyor; `drums` enstrümanı yalnızca
  katalogda ve kanal kuralında var.
- **Tek stil**: düzenleme koral dokusuna göre tasarlandı; caz/elektronik
  istekleri için önce veri (PDMX) gerekiyor.
- **Rol düzenlemesi kural kadar**: roller perde bölgesine göre atanır, motif
  tekrarı/çiftleme/kontrpuan yok. "Melodi şu cümleyi çalsın" gibi istekler için
  gerçek orkestrasyon katmanı (8+) gerekiyor.
