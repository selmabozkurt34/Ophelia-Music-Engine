"""Değerlendirilmiş MIDI kaynakları: lisans kararları ve gerekçeleri.

Bu tablo 3. aşamadaki araştırmanın kalıcı kaydıdır. Üretim hattı yalnızca
``verdict == "approved"`` olan kaynakları kabul eder; ``rejected`` bir kaynağı
kullanmaya çalışmak hata verir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ophelia.data.licensing import get_license

APPROVED = "approved"
REJECTED = "rejected"
REVIEW = "review"


@dataclass(frozen=True)
class Source:
    """Tek bir veri kaynağının değerlendirmesi."""

    slug: str
    name: str
    license_id: str
    url: str
    formats: tuple[str, ...]
    file_count: int | None
    verdict: str
    rationale: str
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def license(self):  # noqa: ANN201 - License döner
        return get_license(self.license_id)

    @property
    def commercial_ok(self) -> bool:
        return self.license.commercial_use and self.verdict != REJECTED

    @property
    def usable(self) -> bool:
        """Üretim (ticari) hattında kullanılabilir mi?"""
        return self.verdict == APPROVED and self.license.commercial_use


SOURCES: dict[str, Source] = {
    source.slug: source
    for source in (
        Source(
            slug="pdmx",
            name="PDMX (Public Domain MusicXML)",
            license_id="CC0-1.0",
            url="https://zenodo.org/records/15571083",
            formats=("musicxml", "midi"),
            file_count=254_077,
            verdict=APPROVED,
            rationale=(
                "Makale veri setini açıkça 'ticari olarak kullanılabilir' diye tanımlıyor; "
                "kaynak dosyalar MuseScore üzerinde PD/CC0 olarak işaretli. En geniş, en "
                "temiz lisans zinciri."
            ),
            caveats=(
                "Lisans tutarsızlığı olan %12,3'lük dilim (license_conflict) kullanılmamalı.",
                "Yalnızca no_license_conflict ve all_valid alt kümeleri alınmalı.",
            ),
        ),
        Source(
            slug="openscore-lieder",
            name="OpenScore Lieder Corpus",
            license_id="CC0-1.0",
            url="https://github.com/OpenScore/Lieder",
            formats=("mscx", "musicxml"),
            file_count=1_300,
            verdict=APPROVED,
            rationale=(
                "Kodlamalar CC0 ile kamu malına bırakılmış; 19. yüzyıl lied repertuvarı "
                "zaten kamu malı. Lisans zinciri en net olan kaynak."
            ),
            caveats=(
                "MuseScore (.mscx) formatında; MIDI'ye dönüştürme adımı gerekir.",
                "Tür olarak dar: vokal + piyano lied.",
            ),
        ),
        Source(
            slug="giantmidi-piano",
            name="GiantMIDI-Piano",
            license_id="CC-BY-4.0",
            url="https://github.com/bytedance/GiantMIDI-Piano",
            formats=("midi",),
            file_count=10_855,
            verdict=REVIEW,
            rationale=(
                "CC BY 4.0 ve repertuvar ağırlıklı kamu malı klasik. Ancak indirme, "
                "okunmamış bir disclaimer.md kabulü gerektiriyor."
            ),
            caveats=(
                "disclaimer.md metni doğrulanmadı; ek koşul getirebilir.",
                "YouTube kayıtlarından otomatik transkripsiyon; 20. yy eserleri karışabilir.",
            ),
        ),
        Source(
            slug="music21-bach-chorales",
            name="Bach Chorales (music21 corpus)",
            license_id="PD",
            url="https://github.com/cuthbertLab/music21",
            formats=("mxl", "midi"),
            file_count=433,
            verdict=APPROVED,
            rationale=(
                "Eserler J.S. Bach'a ait (ö. 1750) ve dünya çapında kamu malı. music21 "
                "paketi BSD-3 ile dağıtılıyor; korale kodlamaları Bach-Gesellschaft "
                "dizgisinden geliyor."
            ),
            caveats=(
                "Kodlamanın kendisi için ayrı bir lisans dosyası yok; kamu malı statüsü "
                "eserin yaşına dayanıyor.",
                "Tek stil (4 sesli korale); çeşitlilik için başka kaynak gerekir.",
            ),
        ),
        Source(
            slug="js-fakes",
            name="js-fakes (AI üretimi koraller)",
            license_id="CC-BY-4.0",
            url="https://github.com/omarperacha/js-fakes",
            formats=("midi",),
            file_count=1_200,
            verdict=APPROVED,
            rationale="Açık CC BY 4.0; 500 etiketli + 700 etiketsiz MIDI dosyası.",
            caveats=(
                "Bach değil, model üretimi; stil çeşitliliği katmaz.",
                "Atıf zorunlu.",
            ),
        ),
        Source(
            slug="lakh",
            name="Lakh MIDI Dataset (LMD)",
            license_id="CC-BY-4.0",
            url="https://colinraffel.com/projects/lmd/",
            formats=("midi",),
            file_count=176_581,
            verdict=REJECTED,
            rationale=(
                "Etikette CC BY 4.0 yazsa da dosyalar internetteki kaynaklardan toplanmış; "
                "telifli popüler şarkılar içeriyor. Lisans veren, eserlerin hak sahibi değil."
            ),
            caveats=("Ticari üretken model eğitimi için hukuken riskli.",),
        ),
        Source(
            slug="maestro",
            name="MAESTRO v3.0.0",
            license_id="CC-BY-NC-SA-4.0",
            url="https://magenta.tensorflow.org/datasets/maestro",
            formats=("midi", "audio"),
            file_count=1_276,
            verdict=REJECTED,
            rationale="Lisans açıkça NonCommercial; ticari kullanım yasak.",
            caveats=("ShareAlike maddesi model ağırlıkları için ayrıca sorun yaratır.",),
        ),
        Source(
            slug="metamidi",
            name="MetaMIDI Dataset",
            license_id="research-only",
            url="https://www.metacreation.net/projects/metamidi-dataset",
            formats=("midi",),
            file_count=436_631,
            verdict=REJECTED,
            rationale="Erişim başvuruyla ve yalnızca araştırma amaçlı; yeniden dağıtım yasak.",
            caveats=("Ardılı GigaMIDI de non-commercial.",),
        ),
        Source(
            slug="gigamidi",
            name="GigaMIDI",
            license_id="research-only",
            url="https://huggingface.co/datasets/Metacreation/GigaMIDI",
            formats=("midi",),
            file_count=1_437_304,
            verdict=REJECTED,
            rationale="Kart metni yalnızca ticari olmayan araştırma/eğitime izin veriyor.",
        ),
        Source(
            slug="essen",
            name="Essen Folksong Collection",
            license_id="research-only",
            url="https://github.com/ccarh/essen-folksong-collection",
            formats=("kern",),
            file_count=6_255,
            verdict=REJECTED,
            rationale=(
                "CCARH lisansı tek kullanıcılı ve akademik araştırmayla sınırlı; ticari "
                "türev ve ses kaydı açıkça yasaklanmış."
            ),
            caveats=("Humdrum kern formatı; MIDI'ye çevrim ayrı adım gerektirir.",),
        ),
        Source(
            slug="nottingham",
            name="Nottingham Dataset",
            license_id="GPL-3.0",
            url="https://github.com/jukedeck/nottingham-dataset",
            formats=("abc", "midi"),
            file_count=1_200,
            verdict=REJECTED,
            rationale=(
                "Temizlenmiş sürüm GPLv3; veri için uygun olmayan copyleft. Üst kaynağın "
                "lisansı ise belirsiz."
            ),
        ),
    )
}


def get_source(slug: str) -> Source:
    """Kaynağı adıyla getirir."""
    try:
        return SOURCES[slug.strip().lower()]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCES))
        raise ValueError(f"bilinmeyen kaynak: {slug!r} (mevcut: {available})") from exc


def approved_sources() -> tuple[Source, ...]:
    return tuple(source for source in SOURCES.values() if source.verdict == APPROVED)


def rejected_sources() -> tuple[Source, ...]:
    return tuple(source for source in SOURCES.values() if source.verdict == REJECTED)


def review_sources() -> tuple[Source, ...]:
    return tuple(source for source in SOURCES.values() if source.verdict == REVIEW)


def assert_usable(slug: str) -> Source:
    """Kaynağı üretim hattında kullanmadan önce lisans kapısı.

    Onaylanmamış ya da ticari kullanıma kapalı bir kaynak için hata verir.
    """
    source = get_source(slug)
    if source.verdict != APPROVED:
        raise PermissionError(
            f"'{source.slug}' kaynağı onaylı değil (durum: {source.verdict}). "
            f"Gerekçe: {source.rationale}"
        )
    if not source.license.commercial_use:
        raise PermissionError(
            f"'{source.slug}' lisansı ({source.license_id}) ticari kullanıma izin vermiyor."
        )
    return source
