# 1. Aşama — Geliştirme Ortamının Hazırlanması

Bu belge, Ophelia Music Engine'in üzerine inşa edildiği çalışma ortamını kayda geçirir.
Amaç: **model geliştirmeye hazır, çalışan bir yazılım ortamı.**

## Donanım (otomatik tespit edildi)

| Bileşen | Değer | Eğitim açısından anlamı |
| --- | --- | --- |
| CPU | Intel Core i3-1315U, 4 çekirdek / 6 iş parçacığı | Küçük modeller, kısa bağlam |
| RAM | 7,6 GB | Toplu iş boyutu (batch) küçük tutulmalı |
| GPU | Intel UHD Graphics (CUDA yok) | **PyTorch CPU sürümü** kullanılır |
| Disk (boş) | ~363 GB | Veri seti + deney kayıtları için yeterli |

**Sonuç:** CUDA bulunmadığından eğitim CPU üzerinde yapılacaktır. Bu, 5. aşamadaki
mimari seçimini doğrudan belirler:

- Parametre sayısı hedefi: **≤ 10 M** (küçük transformer / GRU)
- Bağlam uzunluğu: 512–1024 token
- Eğitim süresi: dakikalar–saatler mertebesinde, kademeli (checkpoint'li)
- CPU'da verimlilik için `torch.set_num_threads(6)` ve `float32` kullanılmalı

## Kurulan yazılımlar

| Yazılım | Sürüm | Not |
| --- | --- | --- |
| Python | 3.11.9 | `winget install Python.Python.3.11` ile kullanıcı kapsamında |
| Git | 2.55.0 | Sistemde mevcuttu |
| VS Code | 1.143.1 | `winget install Microsoft.VisualStudioCode --scope user` |
| PyTorch | 2.15.0+cpu | CPU indeksinden kuruldu (CUDA yükü yok) |
| Node.js | — | Gerekmiyor (9. aşama arayüzü Python tabanlı olacak) |

VS Code, depo içindeki [.vscode/settings.json](../.vscode/settings.json) sayesinde
açılışta doğru yorumlayıcıyı (`.venv`) kullanır ve pytest entegrasyonu hazırdır;
[.vscode/extensions.json](../.vscode/extensions.json) önerilen eklentileri (Python,
Pylance, Ruff, Jupyter) listeler.

Projeyi editörde açmak için:

```bash
"/c/Users/selma/AppData/Local/Programs/Microsoft VS Code/bin/code.cmd" .
```

## Sanal ortam

```powershell
# Etkinleştir (PowerShell)
.\.venv\Scripts\Activate.ps1

# Git Bash
source .venv/Scripts/activate
```

Ortam `C:\Users\selma\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv`
ile oluşturuldu; bağımlılıklar [requirements.txt](../requirements.txt) içinde sabitlenmiştir.

## Kurulum komutları (yeniden üretim için)

```bash
winget install --id Python.Python.3.11 --exact --scope user
winget install --id Microsoft.VisualStudioCode --exact --scope user
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pip install -e .   # `ophelia` komutu
```

## Doğrulama

```bash
./.venv/Scripts/python.exe scripts/check_env.py   # tam ortam raporu
./.venv/Scripts/python.exe -m ophelia env          # aynı rapor, CLI üzerinden
./.venv/Scripts/python.exe -m pytest               # kabul testleri
```

`check_env.py` çıkış kodu 0 ise ortam hazırdır. Ses fontu (SF2) kontrolü 7. aşamaya
kadar **uyarı** niteliğindedir, hata değildir.

## Klasör yapısı

```
Ophelia Music Engine/
├── configs/            # YAML yapılandırmalar (deney başına bir dosya)
├── data/
│   ├── raw/            # Ham MIDI (lisans kaydı ile birlikte)
│   ├── interim/        # Temizlenmiş ara çıktılar
│   ├── processed/      # Modele hazır temsil
│   ├── splits/         # train / val / test bölünmeleri
│   └── manifests/      # Lisans ve kaynak kayıtları (JSON)
├── docs/               # Aşama belgeleri
├── models/             # Checkpoint'ler ve son ağırlıklar
├── ophelia/            # Python paketi
│   ├── core/           # yollar, yapılandırma, günlükleme, ortam denetimi
│   ├── theory/         # 2. aşama
│   ├── data/           # 3-4. aşama
│   ├── model/          # 5. aşama
│   ├── nlp/            # 6. aşama
│   ├── audio/          # 7-8. aşama
│   ├── ui/             # 9. aşama
│   └── cli/            # komut satırı
├── outputs/            # Üretilen MIDI ve WAV dosyaları
├── scripts/            # Yardımcı betikler
└── tests/              # Testler
```

## Bilinen kısıtlar

- **OneDrive:** Proje, OneDrive'a bağlı Masaüstü klasöründe duruyor. `.venv`
  (~1,5 GB) ve veri setleri senkronize edilmeye çalışılır. Büyük veri toplanmaya
  başlandığında projeyi `C:\dev\ophelia` gibi OneDrive dışı bir yola taşımak önerilir.
- **CPU eğitimi:** 6 iş parçacığı, 7,6 GB RAM. Eğitim betikleri bellek dostu olmalı
  (`num_workers=0` veya 2, küçük batch, gradient accumulation).
- **Genişletme:** GPU'lu bir makineye geçilirse yalnızca `pip install torch --index-url
  https://download.pytorch.org/whl/cu124` yeterlidir; kod `torch.cuda.is_available()`
  üzerinden otomatik seçim yapar.
