"""Zero-dependency Gulf & Urban Najdi Clitic Morphological Analyzer (< 10 µs)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .lexicon import NajdiLexicon, get_default_lexicon, light_arabic_key, strip_diacritics

_SUN_LETTERS = frozenset("تثدذرزسشصضطظلن")
_SHORT_VOWELS_SUKUN = frozenset("\u064E\u064F\u0650\u0652")

# Proclitic prefixes for nominal stems: (surface, proclitic_tuple, vocalized_prefix, has_definite_al)
_NOMINAL_PROCLITICS: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    ("وهال", ("و", "هال"), "وَهَا", True),
    ("فهال", ("ف", "هال"), "فَهَا", True),
    ("بهال", ("ب", "هال"), "بِهَا", True),
    ("لهال", ("ل", "هال"), "لِهَا", True),
    ("كهال", ("ك", "هال"), "كَهَا", True),
    ("هال", ("هال",), "هَا", True),
    ("وبال", ("و", "ب", "ال"), "وَبِ", True),
    ("فبال", ("ف", "ب", "ال"), "فَبِ", True),
    ("ولل", ("و", "ل", "ال"), "وَلِلْ", True),
    ("فلل", ("ف", "ل", "ال"), "فَلِلْ", True),
    ("وال", ("و", "ال"), "وَ", True),
    ("فال", ("ف", "ال"), "فَ", True),
    ("بال", ("ب", "ال"), "بِ", True),
    ("كال", ("ك", "ال"), "كَ", True),
    ("لل", ("ل", "ال"), "لِلْ", True),
    ("ال", ("ال",), "", True),
    ("وب", ("و", "ب"), "وَبِ", False),
    ("فب", ("ف", "ب"), "فَبِ", False),
    ("ول", ("و", "ل"), "وَلِ", False),
    ("فل", ("ف", "ل"), "فَلِ", False),
    ("وك", ("و", "ك"), "وَكَ", False),
    ("و", ("و",), "وَ", False),
    ("ف", ("ف",), "فَ", False),
    ("ب", ("ب",), "بِ", False),
    ("ل", ("ل",), "لِ", False),
    ("ك", ("ك",), "كَ", False),
)

# Nominal enclitics (possessive pronouns & plural/dual markers)
# (surface, enclitic_tuple, vocalized_suffix, requires_indefinite, fem_plural_at)
_NOMINAL_ENCLITICS: tuple[tuple[str, tuple[str, ...], str, bool, bool], ...] = (
    ("اتهم", ("ات", "هم"), "َاتْهُمْ", True, True),
    ("اتكم", ("ات", "كم"), "َاتْكُمْ", True, True),
    ("اتهن", ("ات", "هن"), "َاتْهُنْ", True, True),
    ("اتها", ("ات", "ها"), "َاتْهَا", True, True),
    ("اتنا", ("ات", "نا"), "َاتْنَا", True, True),
    ("اتك", ("ات", "ك"), "َاتِكْ", True, True),
    ("اته", ("ات", "ه"), "َاتِهْ", True, True),
    ("اتي", ("ات", "ي"), "َاتِي", True, True),
    ("ات", ("ات",), "َاتْ", False, True),
    ("هم", ("هم",), "هُمْ", True, False),
    ("كم", ("كم",), "كُمْ", True, False),
    ("هن", ("هن",), "هُنْ", True, False),
    ("ها", ("ها",), "هَا", True, False),
    ("نا", ("نا",), "نَا", True, False),
    ("ين", ("ين",), "َيْنْ", False, False),
    ("ون", ("ون",), "ُونْ", False, False),
    ("ك", ("ك",), "ِكْ", True, False),
    ("ه", ("ه",), "ِهْ", True, False),
)

# Imperfective verb stems (after ت/ي/ن/أ): bare_stem -> vocalized_stem (with pausal sukun)
IMPERFECTIVE_STEMS: dict[str, str] = {
    "عرف": "عْرِفْ",
    "قول": "قُولْ",
    "قدر": "قْدَرْ",
    "شوف": "شُوفْ",
    "سمع": "سْمَعْ",
    "كتب": "كْتِبْ",
    "قرا": "قْرَا",
    "فهم": "فْهَمْ",
    "حسب": "حْسِبْ",
    "دخل": "دْخُلْ",
    "طلع": "طْلَعْ",
    "رجع": "رْجَعْ",
    "ضبط": "ضْبِطْ",
    "لحق": "لْحَقْ",
    "صير": "صِيرْ",
    "روح": "رُوحْ",
    "مشي": "مْشِي",
    "جي": "جِي",
    "بي": "بِي",
    "شتغل": "شْتَغِلْ",
    "ستفيد": "سْتَفِيدْ",
    "ستمر": "سْتَمِرّ",
    "نتهي": "نْتَهِي",
    "ختصر": "خْتَصِرْ",
    "عتمد": "عْتَمِدْ",
    "نتظر": "نْتَظِرْ",
    "ستلم": "سْتَلِمْ",
    "ستخدم": "سْتَخْدِمْ",
    "ستفسر": "سْتَفْسِرْ",
    "توقع": "تَوَقَّعْ",
    "تكلم": "تْكَلَّمْ",
    "تأكد": "تَأَكَّدْ",
    "تاكد": "تَأَكَّدْ",
    "تحسن": "تْحَسَّنْ",
    "تغير": "تْغَيَّرْ",
    "تواصل": "تْوَاصَلْ",
    "شيك": "شَيِّكْ",
    "خلص": "خَلِّصْ",
    "جهز": "جَهِّزْ",
    "رتب": "رَتِّبْ",
    "سوي": "سَوِّي",
    "كمل": "كَمِّلْ",
    "حول": "حَوِّلْ",
    "حدث": "حَدِّثْ",
    "حمل": "حَمِّلْ",
    "نزل": "نَزِّلْ",
    "جرب": "جَرِّبْ",
    "وضح": "وَضِّحْ",
    "وفر": "وَفِّرْ",
    "سهل": "سَهِّلْ",
    "ساعد": "سَاعِدْ",
    "راجع": "رَاجِعْ",
    "ناسب": "نَاسِبْ",
    "عالج": "عَالِجْ",
    "تابع": "تَابِعْ",
    "شارك": "شَارِكْ",
    "وافق": "وَافِقْ",
    "حفظ": "حْفَظْ",
    "ربط": "رْبِطْ",
    "ضيف": "ضِيفْ",
    "ختار": "خْتَارْ",
    "لقى": "لْقَى",
    "نفع": "نْفَعْ",
    "ضمن": "ضْمَنْ",
    "فرق": "فْرِقْ",
    "رسل": "رْسِلْ",
    "حجز": "حْجِزْ",
    "طلب": "طْلُبْ",
    "دفع": "دْفَعْ",
    "سحب": "سْحَبْ",
    "فتح": "فْتَحْ",
    "قفل": "قَفِّلْ",
    "شغل": "شَغِّلْ",
    "سجل": "سَجِّلْ",
    "اكد": "أَكِّدْ",
    "أكد": "أَكِّدْ",
    "لغي": "لْغِي",
    "عدل": "عَدِّلْ",
    "جدد": "جَدِّدْ",
}

# Perfective 3ms verb stems -> vocalized 3ms with terminal sukun for suffixation
PERFECTIVE_STEMS: dict[str, str] = {
    "خلص": "خَلَّصْ",
    "شيك": "شَيَّكْ",
    "جرب": "جَرَّبْ",
    "رتب": "رَتَّبْ",
    "جهز": "جَهَّزْ",
    "حدث": "حَدَّثْ",
    "حمل": "حَمَّلْ",
    "نزل": "نَزَّلْ",
    "حول": "حَوَّلْ",
    "كمل": "كَمَّلْ",
    "قفل": "قَفَّلْ",
    "شغل": "شَغَّلْ",
    "وضح": "وَضَّحْ",
    "سجل": "سَجَّلْ",
    "أكد": "أَكَّدْ",
    "اكد": "أَكَّدْ",
    "عدل": "عَدَّلْ",
    "جدد": "جَدَّدْ",
    "اعتمد": "اِعْتَمَدْ",
    "استلم": "اِسْتَلَمْ",
    "انتظر": "اِنْتَظَرْ",
    "اشترك": "اِشْتَرَكْ",
    "اتفق": "اِتَّفَقْ",
    "تأكد": "تَأَكَّدْ",
    "تاكد": "تَأَكَّدْ",
    "تواصل": "تَوَاصَلْ",
    "أرسل": "أَرْسَلْ",
    "ارسل": "أَرْسَلْ",
    "راجع": "رَاجَعْ",
    "تابع": "تَابَعْ",
    "ساعد": "سَاعَدْ",
    "شارك": "شَارَكْ",
    "وافق": "وَافَقْ",
    "فحص": "فَحَصْ",
    "ضبط": "ضَبَطْ",
    "ربط": "رَبَطْ",
    "سحب": "سَحَبْ",
    "دفع": "دَفَعْ",
    "فتح": "فَتَحْ",
    "حجز": "حَجَزْ",
    "طلب": "طَلَبْ",
    "كتب": "كَتَبْ",
    "سمع": "سِمَعْ",
    "فهم": "فِهَمْ",
    "عرف": "عَرَفْ",
    "حفظ": "حَفَظْ",
}

# Imperfective enclitics (plural + object combinations and simple object pronouns)
_IMPERF_ENCLITICS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("ونها", ("ونها",), "ُونْهَا"),
    ("ونهم", ("ونهم",), "ُونْهُمْ"),
    ("ونهن", ("ونهن",), "ُونْهُنْ"),
    ("ونكم", ("ونكم",), "ُونْكُمْ"),
    ("وننا", ("وننا",), "ُونَّا"),
    ("ونه", ("ونه",), "ُونِهْ"),
    ("ونك", ("ونك",), "ُونِكْ"),
    ("وني", ("وني",), "ُونِي"),
    ("ينها", ("ينها",), "ِينْهَا"),
    ("ينهم", ("ينهم",), "ِينْهُمْ"),
    ("ينه", ("ينه",), "ِينِهْ"),
    ("ينك", ("ينك",), "ِينِكْ"),
    ("يني", ("يني",), "ِينِي"),
    ("ون", ("ون",), "ُونْ"),
    ("ين", ("ين",), "ِينْ"),
    ("وا", ("وا",), "ُوا"),
    ("هم", ("هم",), "هُمْ"),
    ("كم", ("كم",), "كُمْ"),
    ("هن", ("هن",), "هُنْ"),
    ("ها", ("ها",), "هَا"),
    ("نا", ("نا",), "نَا"),
    ("ني", ("ني",), "نِي"),
    ("ك", ("ك",), "ِكْ"),
    ("ه", ("ه",), "ِهْ"),
)

# Perfective subject+object compound enclitics
_PERF_ENCLITICS: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    # (surface, tuple, vocalized_suffix, strip_stem_sukun)
    ("توها", ("توها",), "تُوهَا", False),
    ("توهم", ("توهم",), "تُوهُمْ", False),
    ("توهن", ("توهن",), "تُوهُنْ", False),
    ("تونا", ("تونا",), "تُونَا", False),
    ("توني", ("توني",), "تُونِي", False),
    ("توه", ("توه",), "تُوهْ", False),
    ("تيها", ("تيها",), "تِيهَا", False),
    ("تيهم", ("تيهم",), "تِيهُمْ", False),
    ("تيني", ("تيني",), "تِينِي", False),
    ("تيه", ("تيه",), "تِيهْ", False),
    ("ناها", ("نا", "ها"), "ْنَاهَا", True),
    ("ناهم", ("نا", "هم"), "ْنَاهُمْ", True),
    ("ناكم", ("نا", "كم"), "ْنَاكُمْ", True),
    ("ناه", ("نا", "ه"), "ْنَاهْ", True),
    ("تها", ("ت", "ها"), "تْهَا", False),
    ("تهم", ("ت", "هم"), "تْهُمْ", False),
    ("تكم", ("ت", "كم"), "تْكُمْ", False),
    ("تنا", ("ت", "نا"), "تْنَا", False),
    ("تني", ("ت", "ني"), "تْنِي", False),
    ("ته", ("ت", "ه"), "تِهْ", False),
    ("تك", ("ت", "ك"), "تِكْ", False),
    ("وها", ("وا", "ها"), "ُوهَا", True),
    ("وهم", ("وا", "هم"), "ُوهُمْ", True),
    ("وكم", ("وا", "كم"), "ُوكُمْ", True),
    ("ونا", ("وا", "نا"), "ُونَا", True),
    ("وني", ("وا", "ني"), "ُونِي", True),
    ("وه", ("وا", "ه"), "ُوهْ", True),
    ("وك", ("وا", "ك"), "ُوكْ", True),
    ("توا", ("وا",), "تُوا", False),
    ("وا", ("وا",), "ُوا", True),
    ("ها", ("ها",), "َهَا", True),
    ("هم", ("هم",), "َهُمْ", True),
    ("كم", ("كم",), "َكُمْ", True),
    ("هن", ("هن",), "َهُنْ", True),
)


def _ensure_pausal_sukun(voc: str) -> str:
    if not voc:
        return voc
    if voc.endswith("ة"):
        return voc + "ْ"
    return voc


def _attach_definite_al(indef_voc: str) -> str:
    """Attach vocalized `ال` to an indefinite vocalized noun, respecting Arabic sun/moon letters."""
    if not indef_voc:
        return indef_voc
    first_char = indef_voc[0]
    if first_char in _SUN_LETTERS:
        if len(indef_voc) > 1 and indef_voc[1] == "\u0651":
            return "ال" + indef_voc
        return "ال" + first_char + "\u0651" + indef_voc[1:]
    if first_char in ("أ", "إ", "آ", "ا"):
        return "الْ" + indef_voc
    return "الْ" + indef_voc


def _strip_definite_al(def_voc: str) -> str:
    """Derive an indefinite vocalized stem from a vocalized `الـ` entry."""
    if def_voc.startswith("الْ"):
        return def_voc[3:]
    if def_voc.startswith("ال") and len(def_voc) >= 4:
        rest = def_voc[2:]
        if rest[0] in _SUN_LETTERS and len(rest) > 1 and rest[1] == "\u0651":
            return rest[0] + rest[2:]
        return rest
    return def_voc


def _prepare_stem_for_enclitic(indef_voc: str, *, fem_plural_at: bool = False) -> str:
    """Prepare a vocalized nominal stem for suffix attachment (`ـة -> ـتْ` or `ـة -> ـات`)."""
    work = indef_voc
    if work.endswith("ْ"):
        work = work[:-1]
    if fem_plural_at:
        if work.endswith("ة"):
            work = work[:-1]
            if work.endswith("\u064E"):
                work = work[:-1]
        return work
    if work.endswith("ة"):
        return work[:-1] + "تْ"
    return work + "ْ" if work and work[-1] not in _SHORT_VOWELS_SUKUN and work[-1] not in "اويى" else work


@dataclass(frozen=True, slots=True)
class MorphAnalysis:
    """Structured Gulf & Najdi clitic decomposition result."""

    token: str
    plain: str
    proclitics: tuple[str, ...]
    stem: str
    enclitics: tuple[str, ...]
    normalized: str
    vocalized: str | None
    source: str = "unknown"

    @property
    def is_resolved(self) -> bool:
        return self.vocalized is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "plain": self.plain,
            "proclitics": list(self.proclitics),
            "stem": self.stem,
            "enclitics": list(self.enclitics),
            "normalized": self.normalized,
            "vocalized": self.vocalized,
            "source": self.source,
        }


class NajdiMorphAnalyzer:
    """Sub-10µs zero-dependency morphological clitic analyzer for Gulf & White Saudi Arabic."""

    def __init__(self, lexicon: NajdiLexicon | None = None) -> None:
        self.lexicon = lexicon or get_default_lexicon()
        self._indef_nouns: dict[str, tuple[str, str]] = {}  # plain_key -> (canonical_stem, indef_voc)
        self._def_nouns: dict[str, tuple[str, str]] = {}  # plain_key -> (canonical_stem, def_voc_after_ha_or_prep)
        self._perf_stems: dict[str, str] = dict(PERFECTIVE_STEMS)
        self._build_stem_tables()
        self._analyze_cached = lru_cache(maxsize=8192)(self._analyze_uncached)

    def _build_stem_tables(self) -> None:
        for entry in self.lexicon.entries:
            if " " in entry.term:
                continue
            plain_term = strip_diacritics(entry.term)
            light_term = light_arabic_key(plain_term)
            voc = _ensure_pausal_sukun(entry.vocalized)

            # Extract 1st/2nd person perfective verbs in lexicon ending with تْ (e.g. خلّصت -> خَلَّصْتْ)
            if entry.pos.startswith("verb") and plain_term.endswith("ت") and voc.endswith("تْ") and len(plain_term) >= 4:
                base_plain = plain_term[:-1]
                base_voc = voc[:-2] + "ْ" if len(voc) > 2 and voc[-3] != "ْ" else voc[:-2]
                if base_voc and not base_voc.endswith("ْ"):
                    base_voc += "ْ"
                self._perf_stems.setdefault(base_plain, base_voc)
                self._perf_stems.setdefault(light_arabic_key(base_plain), base_voc)

            if plain_term.startswith("ال") and len(plain_term) >= 4:
                bare = plain_term[2:]
                bare_light = light_arabic_key(bare)
                indef_v = _strip_definite_al(voc)
                self._def_nouns.setdefault(bare, (bare, voc))
                self._def_nouns.setdefault(bare_light, (bare, voc))
                self._indef_nouns.setdefault(bare, (bare, indef_v))
                self._indef_nouns.setdefault(bare_light, (bare, indef_v))
            elif len(plain_term) >= 2:
                def_v = _attach_definite_al(voc)
                self._indef_nouns.setdefault(plain_term, (entry.term, voc))
                self._indef_nouns.setdefault(light_term, (entry.term, voc))
                self._def_nouns.setdefault(plain_term, (entry.term, def_v))
                self._def_nouns.setdefault(light_term, (entry.term, def_v))

    def _lookup_nominal(self, stem: str) -> tuple[str, str, str] | None:
        """Return `(canonical_indef, indef_voc, def_voc)` if `stem` is a known nominal stem."""
        hit_indef = self._indef_nouns.get(stem) or self._indef_nouns.get(light_arabic_key(stem))
        if hit_indef is None:
            return None
        hit_def = self._def_nouns.get(stem) or self._def_nouns.get(light_arabic_key(stem))
        def_voc = hit_def[1] if hit_def is not None else _attach_definite_al(hit_indef[1])
        return hit_indef[0], hit_indef[1], def_voc

    def _try_verbal(self, plain: str) -> MorphAnalysis | None:
        """Decompose Gulf/Najdi imperfective (`وبنشيكها`, `بيعتمدونها`) and perfective (`خلصتوها`)."""
        # 1. Imperfective: optional (و|ف) + optional future (ب) + mudara'ah (ت|ي|ن|أ|ا) + stem + enclitic
        conj_plain = ""
        conj_voc = ""
        proclitics: list[str] = []
        work = plain

        if len(work) >= 4 and work[0] in ("و", "ف"):
            conj_plain = work[0]
            conj_voc = "وَ" if conj_plain == "و" else "فَ"
            proclitics.append(conj_plain)
            work = work[1:]

        future_ba = False
        if len(work) >= 4 and work[0] == "ب" and work[1] in ("ت", "ي", "ن", "أ", "ا"):
            future_ba = True
            proclitics.append("ب")
            work = work[1:]

        if len(work) >= 3 and work[0] in ("ت", "ي", "ن", "أ", "ا"):
            pref = work[0]
            rest = work[1:]

            matched_stem: str | None = None
            enclitic_tuple: tuple[str, ...] = ()
            enclitic_voc = ""
            enclitic_surf = ""

            if rest in IMPERFECTIVE_STEMS:
                matched_stem = rest
            else:
                for enc_surf, enc_tup, enc_v in _IMPERF_ENCLITICS:
                    if rest.endswith(enc_surf) and len(rest) - len(enc_surf) >= 2:
                        cand = rest[: -len(enc_surf)]
                        if cand in IMPERFECTIVE_STEMS:
                            matched_stem = cand
                            enclitic_tuple = enc_tup
                            enclitic_voc = enc_v
                            enclitic_surf = enc_surf
                            break

            if matched_stem is not None:
                voc_stem = IMPERFECTIVE_STEMS[matched_stem]
                if pref in ("أ", "ا"):
                    voc_pref = "بَأَ" if future_ba else "أَ"
                    norm_pref = "أ"
                else:
                    # Avoid CCC cluster if stem starts with sukun (e.g. عْتَمِدْ -> بِيِعْتَمِدْ, شَيِّكْ -> بِنْشَيِّكْ)
                    starts_with_cluster = len(voc_stem) > 1 and voc_stem[1] == "\u0652"
                    if future_ba:
                        voc_pref = f"بِ{pref}ِ" if starts_with_cluster else f"بِ{pref}ْ"
                    else:
                        voc_pref = f"{pref}ِ"
                    norm_pref = pref

                combined_voc = f"{conj_voc}{voc_pref}{voc_stem}"
                if enclitic_voc:
                    if enclitic_voc[0] in ("\u064F", "\u0650") and combined_voc.endswith("ْ"):
                        combined_voc = combined_voc[:-1] + enclitic_voc
                    else:
                        combined_voc = combined_voc + enclitic_voc

                norm_surface = f"{conj_plain}{'ب' if future_ba else ''}{norm_pref}{matched_stem}{enclitic_surf}"
                return MorphAnalysis(
                    token=plain,
                    plain=plain,
                    proclitics=tuple(proclitics),
                    stem=f"{norm_pref}{matched_stem}",
                    enclitics=enclitic_tuple,
                    normalized=norm_surface,
                    vocalized=combined_voc,
                    source="morphology",
                )

        # 2. Perfective with compound clitics (`خلصتوها`, `وخلصتوها`, `اعتمدوها`, `جربتيها`)
        p_work = plain
        p_procs: list[str] = []
        p_conj_voc = ""
        p_conj_plain = ""
        if len(p_work) >= 5 and p_work[0] in ("و", "ف"):
            p_conj_plain = p_work[0]
            p_conj_voc = "وَ" if p_conj_plain == "و" else "فَ"
            p_procs.append(p_conj_plain)
            p_work = p_work[1:]

        for enc_surf, enc_tup, enc_v, strip_suk in _PERF_ENCLITICS:
            if p_work.endswith(enc_surf) and len(p_work) - len(enc_surf) >= 3:
                cand = p_work[: -len(enc_surf)]
                voc_perf = self._perf_stems.get(cand) or self._perf_stems.get(light_arabic_key(cand))
                if voc_perf is not None:
                    base_v = voc_perf[:-1] if (strip_suk and voc_perf.endswith("ْ")) else voc_perf
                    full_voc = f"{p_conj_voc}{base_v}{enc_v}"
                    return MorphAnalysis(
                        token=plain,
                        plain=plain,
                        proclitics=tuple(p_procs),
                        stem=cand,
                        enclitics=enc_tup,
                        normalized=f"{p_conj_plain}{cand}{enc_surf}",
                        vocalized=full_voc,
                        source="morphology",
                    )
        return None

    def _try_nominal(self, plain: str) -> MorphAnalysis | None:
        """Decompose Gulf/Najdi nominal proclitics (`هالـ`, `وبـ`, `للـ`) and enclitics (`ـاتهم`, `ـها`)."""
        # Try proclitic + stem (+ optional enclitic) AND bare stem + enclitic
        proc_candidates: list[tuple[str, tuple[str, ...], str, bool]] = [("", (), "", False)]
        for surf, tup, v_pref, has_al in _NOMINAL_PROCLITICS:
            if plain.startswith(surf) and len(plain) - len(surf) >= 2:
                proc_candidates.append((surf, tup, v_pref, has_al))

        # Prefer specific proclitics (like `هال`) before bare/single-letter fallback when stem resolves
        proc_candidates.sort(key=lambda item: len(item[0]), reverse=True)

        for proc_surf, proc_tup, proc_voc, has_al in proc_candidates:
            rem = plain[len(proc_surf) :]
            if len(rem) < 2:
                continue

            # Case A: No enclitic (e.g. `هالموضوع`, `هالشغلة`, `هالفاتورة`, `وبالفاتورة`, `للمكالمة`)
            nom = self._lookup_nominal(rem)
            # Safe final ه -> ة recovery ONLY under definite `ال` / `هال` where pronoun `ـه` is ungrammatical
            if nom is None and has_al and rem.endswith("ه") and len(rem) >= 3:
                nom = self._lookup_nominal(rem[:-1] + "ة")
                if nom is not None:
                    rem = rem[:-1] + "ة"

            if nom is not None and proc_tup:
                canon_stem, indef_voc, def_voc = nom
                if has_al:
                    # `لل` already carries `لِلْ`; drop leading `الْ` or `ال` from `def_voc`
                    if proc_surf.endswith("لل"):
                        after_ll = def_voc[3:] if def_voc.startswith("الْ") else (def_voc[2:] if def_voc.startswith("ال") else def_voc)
                        full_voc = proc_voc + after_ll
                    else:
                        full_voc = proc_voc + def_voc
                else:
                    full_voc = proc_voc + indef_voc

                norm_word = f"{proc_surf}{strip_diacritics(canon_stem)}"
                return MorphAnalysis(
                    token=plain,
                    plain=plain,
                    proclitics=proc_tup,
                    stem=canon_stem,
                    enclitics=(),
                    normalized=norm_word,
                    vocalized=full_voc,
                    source="lexicon",
                )

            # Case B: Stem + enclitic (e.g. `بمكالماتهم`, `فاتورتها`, `هالمكالمات`, `بشغلتهم`)
            for enc_surf, enc_tup, enc_voc, req_indef, is_fem_at in _NOMINAL_ENCLITICS:
                if has_al and req_indef:
                    # Definite `ال`/`هال` cannot co-occur with possessive pronouns, only with plural `ـات`/`ـين`/`ـون`
                    continue
                if not rem.endswith(enc_surf) or len(rem) - len(enc_surf) < 2:
                    continue

                raw_stem = rem[: -len(enc_surf)]
                candidates_to_try: list[str] = [raw_stem]
                if is_fem_at:
                    # `مكالم` + `ات` -> `مكالمة`
                    candidates_to_try.insert(0, raw_stem + "ة")
                elif raw_stem.endswith("ت") and len(raw_stem) >= 3:
                    # `فاتورت` + `ها` -> `فاتورة`
                    candidates_to_try.insert(0, raw_stem[:-1] + "ة")

                for cand_stem in candidates_to_try:
                    nom_hit = self._lookup_nominal(cand_stem)
                    if nom_hit is None:
                        continue
                    canon_stem, indef_voc, def_voc = nom_hit
                    if has_al:
                        # e.g. `هالمكالمات` -> `هَا` + `الْمُكَالَم` + `َاتْ`
                        base_for_al = _prepare_stem_for_enclitic(def_voc, fem_plural_at=is_fem_at)
                        if proc_surf.endswith("لل") and base_for_al.startswith("الْ"):
                            full_voc = proc_voc + base_for_al[3:] + enc_voc
                        elif proc_surf.endswith("لل") and base_for_al.startswith("ال"):
                            full_voc = proc_voc + base_for_al[2:] + enc_voc
                        else:
                            full_voc = proc_voc + base_for_al + enc_voc
                    else:
                        base_v = _prepare_stem_for_enclitic(indef_voc, fem_plural_at=is_fem_at)
                        if enc_voc[0] in ("\u064E", "\u064F", "\u0650") and base_v.endswith("ْ"):
                            full_voc = f"{proc_voc}{base_v[:-1]}{enc_voc}"
                        else:
                            full_voc = f"{proc_voc}{base_v}{enc_voc}"

                    return MorphAnalysis(
                        token=plain,
                        plain=plain,
                        proclitics=proc_tup,
                        stem=canon_stem,
                        enclitics=enc_tup,
                        normalized=f"{proc_surf}{rem}",
                        vocalized=full_voc,
                        source="morphology",
                    )

        return None

    def _analyze_uncached(self, token: str) -> MorphAnalysis:
        clean = token.strip()
        plain = strip_diacritics(clean)
        if not plain:
            return MorphAnalysis(
                token=token,
                plain="",
                proclitics=(),
                stem="",
                enclitics=(),
                normalized=token,
                vocalized=None,
            )

        # Productive `هالـ` demonstratives and compound verbal clitics decompose structurally first
        if plain.startswith(("هال", "وهال", "بهال", "لهال", "فهال", "كهال")):
            nom_ha = self._try_nominal(plain)
            if nom_ha is not None:
                return nom_ha

        if plain.endswith(("توها", "توهم", "تيها", "ونها", "ونهم", "ونه")):
            verb_comp = self._try_verbal(plain)
            if verb_comp is not None:
                return verb_comp

        # 1. Direct lexicon hit
        direct_voc = self.lexicon.vocalize_token(clean)
        if direct_voc is not None:
            norm = self.lexicon.normalize_token(clean)
            return MorphAnalysis(
                token=token,
                plain=plain,
                proclitics=(),
                stem=norm,
                enclitics=(),
                normalized=norm,
                vocalized=direct_voc,
                source="lexicon",
            )

        # 2. Verbal clitic + stem + suffix decomposition (high specificity)
        verbal = self._try_verbal(plain)
        if verbal is not None:
            return verbal

        # 3. Nominal clitic (`هالـ`, `وبـ`, `للـ`) + stem + plural/possessive decomposition
        nominal = self._try_nominal(plain)
        if nominal is not None:
            return nominal

        # 4. Proclitic + colloquial orthographic variant in lexicon (e.g. `وذلحين` -> `والحين`)
        vmap = self.lexicon.variant_map
        for proc_s in ("و", "ف", "ب", "ل"):
            if plain.startswith(proc_s) and len(plain) >= 4:
                rem = plain[1:]
                canon_rem = vmap.get(rem)
                if canon_rem is not None and " " not in canon_rem:
                    rem_voc = self.lexicon.vocalize_token(canon_rem)
                    p_voc = {"و": "وَ", "ف": "فَ", "ب": "بِ", "ل": "لِ"}[proc_s]
                    return MorphAnalysis(
                        token=token,
                        plain=plain,
                        proclitics=(proc_s,),
                        stem=canon_rem,
                        enclitics=(),
                        normalized=f"{proc_s}{canon_rem}",
                        vocalized=f"{p_voc}{rem_voc}" if rem_voc else None,
                        source="lexicon",
                    )

        return MorphAnalysis(
            token=token,
            plain=plain,
            proclitics=(),
            stem=plain,
            enclitics=(),
            normalized=clean,
            vocalized=None,
            source="unknown",
        )

    def analyze(self, token: str) -> MorphAnalysis:
        """Analyze a Gulf/Najdi token into `(proclitics, stem, enclitics, vocalized)` in < 10 µs."""
        res = self._analyze_cached(token)
        if res.token == token:
            return res
        return MorphAnalysis(
            token=token,
            plain=res.plain,
            proclitics=res.proclitics,
            stem=res.stem,
            enclitics=res.enclitics,
            normalized=res.normalized,
            vocalized=res.vocalized,
            source=res.source,
        )

    def vocalize_inflected(self, token: str) -> str | None:
        """Return vocalized form for a direct or clitic-inflected Gulf/Najdi word, or None."""
        return self._analyze_cached(token).vocalized

    def normalize_inflected(self, token: str) -> str:
        """Return orthographically normalized form for direct or clitic-attached Saudi tokens."""
        direct = self.lexicon.variant_map.get(token)
        if direct is not None:
            return direct
        plain = strip_diacritics(token)
        direct_plain = self.lexicon.variant_map.get(plain)
        if direct_plain is not None:
            return direct_plain
        res = self._analyze_cached(token)
        # Only rewrite surface spelling when stem variant or definite `هالـ` taa-marbuta fixed
        if res.source == "lexicon" and res.normalized != plain and strip_diacritics(res.normalized) != plain:
            return res.normalized
        return token


_DEFAULT_MORPH: NajdiMorphAnalyzer | None = None


def get_default_morph_analyzer(lexicon: NajdiLexicon | None = None) -> NajdiMorphAnalyzer:
    """Return cached NajdiMorphAnalyzer singleton."""
    global _DEFAULT_MORPH
    if lexicon is not None:
        return NajdiMorphAnalyzer(lexicon=lexicon)
    if _DEFAULT_MORPH is None:
        _DEFAULT_MORPH = NajdiMorphAnalyzer()
    return _DEFAULT_MORPH
