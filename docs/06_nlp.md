# 6. Aşama — Doğal Dil Komut Sistemi

Kullanıcı ne istediğini kendi cümlesiyle söyler, motor eseri üretir:

```bash
ophelia compose "hüzünlü bir la minör parça, yavaş, 30 saniye"
```

```
  tonalite istenen : A natural_minor
  modelin bulduğu  : A natural_minor
  tempo            : 66 BPM
  nota · süre      : 122 · 30.0 sn
  gam uyumu        : 100% (A natural_minor)
```

## 1. Hat

```
cümle ──parse──► CompositionSpec ──seed──► model ──arrange──► MIDI
        (kural)   (doğrulanmış istek)  (tohum)         (aktarım,
                                                        projeksiyon,
                                                        kırpma)
```

| Modül | İş |
| --- | --- |
| [parse.py](../ophelia/nlp/parse.py) | Türkçe/İngilizce cümle → istek |
| [spec.py](../ophelia/nlp/spec.py) | `CompositionSpec`: tonalite, tempo, süre, polifoni, tohum, enstrüman, rol |
| [seeding.py](../ophelia/nlp/seeding.py) | İstenen tonalitede başlangıç motifi (prompt) |
| [arrange.py](../ophelia/nlp/arrange.py) | Aktarım, gama oturtma, süreye kırpma |
| [compose.py](../ophelia/nlp/compose.py) | Sıra + rapor |

Doğal dil ile model arasındaki tek sözleşme `CompositionSpec`. Bu yüzden komut
çözümlemesi model olmadan test edilebiliyor ve ileride komut kaynağı değişse
(ses, arayüz, LLM) hat aynı kalıyor.

## 2. Çözümleme — kural tabanlı ve deterministik

Ağ yok, LLM yok: aynı cümle her zaman aynı isteği üretir.

| Alan | Anlaşılan kalıplar |
| --- | --- |
| Tonalite | `la minör`, `A minor`, `am`, `fa diyez minör`, `Bb major`, `si bemol majör` |
| Tempo | `120 bpm`, `90 tempo`, `yavaş` (66), `andante` (76), `hızlı` (132), `presto` (168) |
| Süre | `30 saniye`, `45 sn`, `1 dakika`, `8 ölçü`, `kısa` (25 sn), `uzun` (90 sn) |
| Doku | `solo`/`tek ses` (1), `dört ses` (4), `dolu`/`kalın` (6) |
| Ruh hâli | `hüzünlü`, `dramatik`, `neşeli`, `enerjik`, `sakin`, `karanlık` |
| Enstrüman | `keman`, `kontrbas`, `yaylılar`, `piyano`, `flüt`, `davul` |
| Rol (düzenleme) | `melodi`/`ezgi`, `armoni`/`eşlik`, `bas hattı`, `ritim`, `doku` |

**Rol ≠ enstrüman.** `bas` tek başına bas gitarı seçer; cümlede başka bir rol
varsa (`melodi, armoni ve bas`) bas hem enstrüman hem **üçüncü sesin rolü**
olur. Roller `CompositionSpec.roles` alanında kanonik İngilizce adlarla
(`melody`, `harmony`, `bass`) saklanır ve düzenlemeye öyle geçer.

Türkçe karakterler sadeleştirilir (`Hüzünlü, La Minör!` → `huzunlu la minor`),
böylece kullanıcı aksan yazmak zorunda değil.

**Öncelik kuralı:** açıkça yazılan değer her zaman kazanır. Ruh hâli yalnızca
boş alanları doldurur: `hüzünlü` → minör + 68 BPM, `neşeli` → majör + 126 BPM.
Tonalite hiç verilmezse ruh hâli makul bir varsayılan seçer (minör → la minör,
majör → do majör) ve bunu `--explain` çıktısında **söyler**.

`--explain` her kararın gerekçesini gösterir:

```
  • tonalite: 'LA MINOR' → A natural_minor
  • tempo: 'yavas' → 66 BPM
  • süre: '30 saniye' → 30 sn
```

## 3. Modeli yönlendirmek

Model koşulsuz eğitildi: "la minör yaz" diyemez. Üç katman birlikte çalışır.

**1. Tohum.** Üretim, istenen toniğin triadıyla başlar (kök–üçlü–beşli, dörtlük
notalar). Dil modellerindeki prompt'un müzik karşılığı: model hem tonaliteyi
hem dokuyu sezer. Tohum bir **eğilim**dir, garanti değil.

**2. Devam.** Model 5–15 saniyede "bitirebilirim" diyor (`EOS`). İstenen süre
dolmadıkça `EOS` akışa yazılmaz, üretim aynı akışın devamı olarak sürer
(en fazla 4 geçiş). Bu olmadan "30 saniye" isteği 11 saniyelik bir eser veriyordu.

**3. Düzenleme.** Üretim bittiğinde istek hâlâ bir sözdür:

- **Aktarım**: eser, modelin çaldığı tonikten istenen toniğe taşınır (en kısa
  yol: ±6 yarım ton sınırı, aralık dışına çıkan notalar oktavla geri çekilir).
  Aktarım **aralıkları** korur, **modu** değiştirmez — la minör bir ezgiyi
  do'dan çalmak do minör verir. Bu yüzden mod uyuşmazlığı raporda açıkça yazar.
- **Diyatonik projeksiyon**: istenen gamın dışında kalan perdeler en yakın gam
  sesine çekilir (en fazla bir yarım ton). Mod garantisi buradan gelir.
- **Kırpma**: istenen süreyi aşan kısım atılır; sınırı kesen nota kısaltılır,
  ortada kesilmez.
- **Polifoni**: istekte doku belirtildiyse sınır uygulanır ("solo", "dört ses",
  "dolu"). Belirtilmediyse modelin doğal dokusu korunur — "hüzünlü" sözcüğü
  yüzünden notaları kesmek müziği delik deşik ediyordu (185 notanın 85'i
  gidiyordu). Aynı perdenin üst üste başlatılması ise her durumda önlenir:
  önceki nota **kısaltılır**, atılmaz.

## 4. Sonuçlar

`models/ophelia-v3` ile, `--seed 4`:

| Komut | İstenen | Modelin bulduğu | Aktarım | Gam uyumu | Sonuç |
| --- | --- | --- | --- | --- | --- |
| `hüzünlü la minör, yavaş, 30 saniye` | A minör · 66 BPM | A minör | 0 | %100 | 165 nota · 30,0 sn |
| `neşeli do majör 120 bpm 25 saniye` | C majör · 120 BPM | A minör | +3 | %100 | 248 nota · 25,0 sn |
| `dramatik mi bemol minör 90 bpm 20 saniye` | Eb minör · 90 BPM | C majör | +3 | %100 | 140 nota · 20,0 sn |

Üçünde de süre tam tutuyor (kırpma sınırda çalışıyor) ve perdelerin **%100'ü**
istenen gamda. Tonalite tahmini (`estimate_key`) bilgi amaçlı raporlanır; garanti
gam uyumudur çünkü Krumhansl-Kessler tahmini kısa/eser içi parçalarda kayabiliyor
(C majör parçada "G major" diyebiliyor).

## 5. Bilinen sınırlar

- **Anlama sınırı kural kadar**: "Bach gibi ama daha hüzünlü" gibi cümleler
  çözümlenmez; tanınan sözcük yoksa varsayılan istek üretilir ve bu söylenir.
  Serbest metin anlamak 6. aşamanın *bir sonraki* adımı (LLM katmanı).
- **Ruh hâli = tempo + mod**: "hüzün" yalnızca minör/ yavaş demek; melodi
  karakteri, artikülasyon, armonik yoğunluk modellenmiyor.
- **Tek stil**: model koralleri taklit ediyor; "caz" ya da "elektronik" istekleri
  çözümlenir ama üretilen müzik değişmez — bunun için veri (PDMX) gerekli.
- **Enstrüman** komutu tanınmıyor (7. aşama: çok kanallı düzenleme).
- Tohum etkisi garanti değil: model farklı tonda başlarsa düzeltme aktarımla
  yapılır; aktarım melodik olarak duyulabilir (transpoze edilmiş his).
