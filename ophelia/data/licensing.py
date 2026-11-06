"""Lisans modeli: bir veri kaynağının ticari kullanıma uygun olup olmadığını kodlar.

Kural: **Veri seti lisansı, derlemeyi ve MIDI/MusicXML kodlamasını kapsar; altındaki
eserin telif hakkını kapsamaz.** Bu yüzden her kaynak için hem lisans kimliği hem de
insan tarafından yazılmış gerekçe saklanır (bkz. :mod:`ophelia.data.sources`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class License:
    """Tek bir lisans tanımı (SPDX benzeri kimlik + ticari kullanım bilgisi)."""

    id: str
    name: str
    commercial_use: bool
    attribution_required: bool
    share_alike: bool
    url: str
    summary: str

    @property
    def is_open(self) -> bool:
        """Açık lisans mı (ticari kullanıma izin veren)?"""
        return self.commercial_use and not self.share_alike

    def attribution_text(self, source_name: str) -> str:
        """Yayınlanan eserlerde kullanılacak atıf metni."""
        if not self.attribution_required:
            return f"{source_name} (atıf gerekmiyor)"
        return f"{source_name} — {self.name} ({self.url})"


LICENSES: dict[str, License] = {
    "CC0-1.0": License(
        "CC0-1.0",
        "Creative Commons Zero 1.0",
        commercial_use=True,
        attribution_required=False,
        share_alike=False,
        url="https://creativecommons.org/publicdomain/zero/1.0/",
        summary="Telif hakkından feragat; ticari kullanım serbest, atıf zorunlu değil.",
    ),
    "PD": License(
        "PD",
        "Kamu malı (Public Domain)",
        commercial_use=True,
        attribution_required=False,
        share_alike=False,
        url="https://creativecommons.org/publicdomain/mark/1.0/",
        summary=(
            "Eser telif süresi dolduğu için kamu malı. Kodlama/derleme ayrıca "
            "değerlendirilmelidir."
        ),
    ),
    "CC-BY-4.0": License(
        "CC-BY-4.0",
        "Creative Commons Attribution 4.0",
        commercial_use=True,
        attribution_required=True,
        share_alike=False,
        url="https://creativecommons.org/licenses/by/4.0/",
        summary="Ticari kullanım serbest, atıf zorunlu.",
    ),
    "CC-BY-SA-4.0": License(
        "CC-BY-SA-4.0",
        "Creative Commons Attribution ShareAlike 4.0",
        commercial_use=True,
        attribution_required=True,
        share_alike=True,
        url="https://creativecommons.org/licenses/by-sa/4.0/",
        summary="Ticari kullanım serbest ama türevler aynı lisansla paylaşılmalı.",
    ),
    "CC-BY-NC-SA-4.0": License(
        "CC-BY-NC-SA-4.0",
        "Creative Commons Attribution NonCommercial ShareAlike 4.0",
        commercial_use=False,
        attribution_required=True,
        share_alike=True,
        url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        summary="Ticari kullanım YASAK. Yalnızca araştırma/eğitim.",
    ),
    "GPL-3.0": License(
        "GPL-3.0",
        "GNU General Public License v3",
        commercial_use=True,
        attribution_required=True,
        share_alike=True,
        url="https://www.gnu.org/licenses/gpl-3.0.html",
        summary=(
            "Yazılım lisansı; veri için uygun değil. Copyleft, ticari model için riskli."
        ),
    ),
    "research-only": License(
        "research-only",
        "Yalnızca araştırma (özel koşullar)",
        commercial_use=False,
        attribution_required=True,
        share_alike=False,
        url="",
        summary="Erişim kapılı, ticari kullanım ve yeniden dağıtım yasak.",
    ),
    "unknown": License(
        "unknown",
        "Bilinmiyor",
        commercial_use=False,
        attribution_required=True,
        share_alike=False,
        url="",
        summary="Lisansı doğrulanmamış kaynak; üretim hattına alınmamalı.",
    ),
}


def get_license(license_id: str) -> License:
    """Lisansı kimliğiyle getirir."""
    try:
        return LICENSES[license_id.strip()]
    except KeyError as exc:
        available = ", ".join(sorted(LICENSES))
        raise ValueError(f"bilinmeyen lisans: {license_id!r} (mevcut: {available})") from exc


def is_commercially_usable(license_id: str) -> bool:
    """Lisans ticari kullanıma izin veriyor mu?"""
    return get_license(license_id).commercial_use
