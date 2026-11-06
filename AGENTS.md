# Ajan Notları — Ophelia Music Engine

## Otomatik commit + push kuralı (kullanıcı talimatı)

Kullanıcı bir aşamayı bitirmeyi istediğinde ("sonraki aşamaya geç", "aşamayı bitir",
"devam et" gibi), o aşamanın işi biter bitmez **ayrıca sormadan** şunları yap:

1. `ruff check .` ve `pytest` çalıştır. İkisi de temiz değilse push yapma; hatayı düzelt.
2. Değişen dosyaları `git add` ile ekle (yol bazlı; `git add -A` kullanma).
3. Aşamayı anlatan **tek** commit at: Türkçe, "neden" odaklı mesaj + Codebuff imzası.
4. `git push origin main` yap, çıktıyı özetle.

Depo: <https://github.com/selmabozkurt34/Ophelia-Music-Engine>

Bu izin kalıcıdır; her aşamada yeniden sorulmaz. Kuralın **dışında** kalanlar:
force-push, geçmiş değiştirme (`rebase`/`amend` ile uzak geçmişi ezme), dal silme,
depo görünürlüğü değiştirme — bunlar için yine açık onay gerekir.

## Depo düzeni

- `models/`, `outputs/`, `data/raw|interim|processed|splits|tokens` sürüm kontrolünde
  **değil**; boru hattı yeniden üretir. `data/manifests/sources.json` (lisans kaydı)
  ve `.gitkeep` dosyaları izlenir.
- Model ağırlıkları (`best.pt`) depoya girmez; `ophelia model train` ile üretilir.
- Testler: `python -m pytest`. `pyproject.toml` içinde `addopts = "-q"` olduğu için
  özet satırını görmek istersen `python -m pytest -o addopts="" -q`.
- Biçim kapısı `ruff check .`; `ruff format` bu projede uygulanmıyor (depo o biçimde
  yazılmadı), çalıştırıp toplu biçimlendirme yapma.
