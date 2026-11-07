# Kör A/B dinleme testi

Piyasa raporunun (§7.3) ikinci **P0** maddesi: *"Dinleyici ayırt edemiyor"* iddiasının
ölçülmesi. Rapordaki en büyük ölçüm boşluğuydu — bu belge onu kapatan altyapının
protokolünü, sınırlarını ve sonuçların nasıl okunacağını anlatır.

## 1. Ne ölçülüyor

İki ayrı soru var, ikisi de aynı oturumdan çıkar:

| Soru | Ölçüt | Neden önemli |
| --- | --- | --- |
| Hangisi **daha iyi**? | tercih oranı + %95 Wilson aralığı + kesin binom p | model "kulağa hoş geliyor" mu |
| Hangisi **insan**? | tanıma isabeti + aynı istatistikler | model ayırt edilebilir mi (Turing benzeri) |

İkisi farklı şey söyleyebilir: dinleyici gerçek korali daha çok beğenirken hangisinin
insan olduğunu bilmeyebilir. Raporda ikisi ayrı satır olarak verilir.

## 2. Kurulum — adil karşılaştırma

```bash
ophelia abtest build -n 10 --seconds 20 --seed 42
ophelia abtest serve --dir outputs/abtest/<oturum>        # yalnız bu bilgisayar
ophelia abtest serve --dir outputs/abtest/<oturum> --lan  # yerel ağdaki diğer bilgisayarlar
ophelia abtest score --dir outputs/abtest/<oturum>
```

Kurulumda uygulanan kurallar (hepsi testli):

1. **Gerçek taraf test bölütünden seçilir** (`data/splits/splits.json` → `splits.test`,
   38 koral). Model bu dosyaları eğitimde görmedi; aksi hâlde "model kendi ezberini
   mi okuyor" sorusu ayrışmaz.
2. **İki kaynak aynı hattan geçer**: `arrange(chorale_ensemble(parts))` + `render_parts`.
   Enstrüman, sentez, süre kırpma/doldurma ve tepe normalizasyonu aynı; tek fark notalar.
3. **Süre eşitlenir**: her iki kayıt aynı `seconds` değerine kırpılır ya da sessizle
   doldurulur, böylece "uzun olan daha iyi" yanlılığı oluşmaz.
4. **Taraf ve sıra tohumla rastgele**: aynı `--seed` aynı eşlemeyi verir (tekrar üretilebilir).
5. **Dosya adları opak** (`t01_a.wav`), etiketler yalnız `key.json`'da. Sunucu `key.json`,
   `votes.jsonl` ve `../` yollarını **404** ile reddeder; sayfa etiket taşımaz.

## 3. Dinleyici kimliği

- Kimlik tarayıcıda `localStorage["ophelia.listener"]` içinde (`L-<16 hex>`), sunucuda
  hesap yok.
- **Her kişi kendi bilgisayarından** girerse kimlikler kendiliğinden ayrışır.
- Aynı bilgisayarda ikinci kişi için: `http://<adres>:8823/?yeni=1` ya da sayfadaki
  **"Başka bir dinleyici olarak başla"** düğmesi. Sayfa kimlik kodunu gösterir; iki
  kişinin ayrı kodla girdiği böylece gözle doğrulanır.
- **Bir dinleyici bir denemeye bir oy verir.** Aynı dinleyici aynı denemeyi ikinci kez
  oylarsa `summarize` yalnızca **ilkini** sayar ve raporda `yinelenen oy` olarak
  gösterir. Aksi hâlde aynı kişi testi tekrarlayarak örneklemi şişirir
  (pseudo-replication).

## 4. Yerel ağda dinletme

`--lan` sunucuyu `0.0.0.0` üzerinde açar ve makinenin ağ adreslerini yazar
(`http://10.x.x.x:8823/` gibi). Gerekenler:

- Dinleyiciler **aynı ağda** olmalı (aynı Wi-Fi/kurumsal ağ). Farklı ağdaysa tünel
  gerekir; bu altyapı tünel içermez.
- Windows ilk bağlantıda güvenlik duvarı izni ister; `python.exe` için gelen (inbound)
  kural yoksa bağlantı sessizce düşer.
- Sayfada **kimlik doğrulama yoktur**: adresi bilen aynı ağdaki herkes oy verebilir.
  Bu yüzden `--lan` bilinçli bir tercih; varsayılan `127.0.0.1`.

## 5. Örneklem

`required_votes()` en kötü durumu (p = 0,5) verir:

| Hedef kesinlik | Gereken kararlı oy | Kabaca dinleyici (10 deneme/kişi) |
| --- | --- | --- |
| ±15 puan | 43 | 5 |
| ±10 puan | 97 | 10 |
| ±5 puan | 385 | 39 |

6 denemelik bir oturumda 2 dinleyici ≈ 12 oy → aralık ±25 puan. Bu **pilot**tur;
"gerçek daha iyi" ya da "model daha iyi" iddiası bu büyüklükte kurulamaz.

## 6. Bilinen konfoundlar (dürüst sınırlar)

1. **Seviye eşitlemesi LUFS değil, tepe normalizasyonu.** İki kayıt tepeye göre
   eşitleniyor; algılanan yükseklik farklı olabilir ve tercihi etkileyebilir.
   8. aşamadaki LUFS hedefi bu konfoundu kapatacak (bkz. `docs/08_mixing.md`).
2. **Timbre aynı ve sentetik.** İki kayıt da aynı harmonik sentezden geçiyor. Yani test
   *kompozisyonu* ölçer, *ses kalitesini* değil — "hangi piyano" değil "hangi nota".
3. **Tek stil.** Gerçek taraf yalnızca Bach koralı; sonuç başka stillere genellenemez.
4. **Küçük ve kolay örneklem.** Dinleyiciler büyük olasılıkla geliştiriciye yakın kişiler;
   seçim yanlılığı var. Rapor ederken kimlerin dinlediği yazılmalı.
5. **Sıra etkisi.** Her denemede A/B tarafı rastgele ama dinleyici hep A'yı önce dinliyor;
   ilk dinlenen kayıt avantajı (order bias) ölçülmüyor.

## 7. Sonuçların okunması

`score` çıktısındaki iki yorum satırı kurala bağlıdır:

- %95 aralığı **%50'yi içeriyorsa** → "ayırt edilemedi". Bu bir başarısızlık değil;
  küçük modelin ürün iddiası için yeterli olduğunun kanıtı olabilir (ya da örneklemin
  küçüklüğünün).
- Aralık tamamen %50'nin üstündeyse → "gerçek koral anlamlı biçimde tercih edildi".
- Tanıma tarafında aralık %50'nin üstündeyse → dinleyici insanı şanstan iyi tanıyor,
  yani üretim hâlâ "yapay" duruyor.

Ölçülen ilk oturum (6 deneme, 1 dinleyici, 6 geçerli oy): tercih %33,3 (GA 9,7–70,0,
p = 0,69), tanıma 2/5. Yani henüz iki yönde de iddia yok.

## 8. Dosyalar

| Dosya | İçerik |
| --- | --- |
| `key.json` | gerçek taraf ve kaynak dosya adları — **dinleyiciye gitmez** |
| `trials.json` | dinleyiciye açık liste (`id`, `a`, `b`) |
| `audio/*.wav` | opak adlı kayıtlar |
| `index.html` | dinleme sayfası (etiket içermez) |
| `votes.jsonl` | her oy bir satır: deneme, tercih, tahmin, güven, süre, kimlik |

`votes.jsonl` ekleme (append) ile büyür; oturum silinmedikçe oylar korunur.
