# Ophelia Music Engine - gelistirme ortamini sifirdan kurar (Windows / PowerShell).
# Kullanim:  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Not: Dosya bilincli olarak ASCII tutulur; Windows PowerShell 5.1 BOM'suz
#      UTF-8 dosyalari ANSI olarak okudugu icin Turkce karakterler bozulabilir.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== Ophelia Music Engine - ortam kurulumu ==" -ForegroundColor Cyan

# 1) Python 3.11
$python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Python 3.11 bulunamadi, winget ile kuruluyor..." -ForegroundColor Yellow
    winget install --id Python.Python.3.11 --exact --silent --scope user --accept-package-agreements --accept-source-agreements
}

if (-not (Test-Path $python)) {
    throw "Python 3.11 kurulamadi. https://www.python.org/downloads/ adresinden elle kurun."
}
Write-Host "Python: $(& $python --version)"

# 2) Sanal ortam
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Sanal ortam olusturuluyor (.venv)..." -ForegroundColor Yellow
    & $python -m venv .venv
}

# 3) Bagimliliklar
& $venvPython -m pip install --upgrade pip setuptools wheel
Write-Host "PyTorch (CPU) kuruluyor..." -ForegroundColor Yellow
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $venvPython -m pip install -r requirements-dev.txt
& $venvPython -m pip install -e .

# 4) Dogrulama
& $venvPython scripts/check_env.py
if ($LASTEXITCODE -ne 0) { throw "Ortam dogrulamasi basarisiz oldu." }

& $venvPython -m pytest
if ($LASTEXITCODE -ne 0) { throw "Testler basarisiz oldu." }

Write-Host ""
Write-Host "Ortam hazir. Etkinlestirmek icin:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
