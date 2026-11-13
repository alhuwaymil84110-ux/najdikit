"""High-speed O(1) lexicon & compiled Aho-Corasick / Radix engine for Modern White Saudi Arabic."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from .packs import (
    DialectPack,
    DialectPackRegistry,
    PhraseRadixTrie,
    TokenAhoCorasick,
    get_default_pack_registry,
)

_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_ALEF_RE = re.compile(r"[أإآٱ]")
_DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "white_saudi_lexicon.json"


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel and tatweel characters."""
    return _DIACRITICS_RE.sub("", text)


def light_arabic_key(text: str) -> str:
    """Create a fast lookup key normalizing hamza-on-alef and stripping diacritics."""
    cleaned = _DIACRITICS_RE.sub("", text.strip())
    return _ALEF_RE.sub("ا", cleaned)


def _is_pure_orthographic_variant(variant: str, canonical: str) -> bool:
    """Return True only if `variant` is an orthographic spelling variant of `canonical`,
    not a different morphological inflection, tense, or definiteness state.
    """
    v_plain = strip_diacritics(variant).replace(" ", "")
    c_plain = strip_diacritics(canonical).replace(" ", "")
    if not v_plain or not c_plain or v_plain == c_plain:
        return False

    explicit_pairs = {
        ("ذلحين", "الحين"),
        ("ذالحين", "الحين"),
        ("هالحين", "الحين"),
        ("علشان", "عشان"),
        ("اشلون", "وشلون"),
        ("اشلونك", "وشلونك"),
        ("ويش", "وش"),
        ("كيذا", "كذا"),
    }
    if (v_plain, c_plain) in explicit_pairs:
        return True

    if v_plain.startswith("ال") != c_plain.startswith("ال"):
        return False

    def _skeleton(s: str) -> str:
        s = _ALEF_RE.sub("ا", s)
        s = s.replace("ة", "ه").replace("ى", "ا").replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
        out: list[str] = []
        for ch in s:
            if not out or out[-1] != ch:
                out.append(ch)
        return "".join(out)

    return _skeleton(v_plain) == _skeleton(c_plain)


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    """A single lexical or phrasal entry in the NajdiKit atlas."""

    id: str
    term: str
    vocalized: str
    ipa_hint: str
    category: str
    pos: str
    register: str
    meaning_ar: str
    meaning_en: str
    variants: tuple[str, ...]
    example: str
    pack_id: str = "core"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variants"] = list(self.variants)
        return data


@dataclass(frozen=True, slots=True)
class AntiConfusionRule:
    """Maps non-Saudi dialectal or stiff MSA expressions to natural White Saudi."""

    non_saudi: str
    origin: str
    saudi_white: str
    alt: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "non_saudi": self.non_saudi,
            "origin": self.origin,
            "saudi_white": self.saudi_white,
            "alt": list(self.alt),
            "note": self.note,
        }


class NajdiLexicon:
    """Indexed, zero-dependency lexicon with compiled Radix/Aho-Corasick engines & modular packs."""

    def __init__(
        self,
        custom_lexicon_path: str | os.PathLike[str] | None = None,
        *,
        base_lexicon_path: str | os.PathLike[str] | None = None,
        pack_registry: DialectPackRegistry | None = None,
        include_packs: bool = True,
        extra_packs: Sequence[DialectPack] | None = None,
    ) -> None:
        data_path = Path(base_lexicon_path) if base_lexicon_path else _DEFAULT_DATA_PATH
        with data_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        self.metadata: dict[str, Any] = dict(raw.get("metadata", {}))
        self._entries: list[LexiconEntry] = []
        self._seen_terms: set[str] = set()
        self._exact_map: dict[str, LexiconEntry] = {}
        self._light_map: dict[str, LexiconEntry] = {}
        self._vocalize_map: dict[str, str] = {}
        self._variant_to_canonical: dict[str, str] = {}
        self._multiword_by_first: dict[str, list[tuple[tuple[str, ...], LexiconEntry]]] = {}
        self._phrase_trie: PhraseRadixTrie[LexiconEntry] = PhraseRadixTrie()
        self._loaded_pack_ids: list[str] = ["core"]

        for item in raw.get("entries", []):
            self._register_raw_entry(item, pack_id="core")

        self._anti_confusion: list[AntiConfusionRule] = []
        self._anti_confusion_index: dict[str, AntiConfusionRule] = {}
        for r in raw.get("anti_confusion_map", []):
            self._add_anti_confusion_rule(r)

        self._hallucination_blacklist: list[str] = list(raw.get("hallucination_blacklist", []))

        if include_packs:
            reg = pack_registry or get_default_pack_registry()
            for pack in reg.snapshot():
                self.apply_pack(pack)
        if extra_packs:
            for pack in extra_packs:
                self.apply_pack(pack)

        env_custom = os.environ.get("NAJDIKIT_CUSTOM_LEXICON", "").strip()
        effective_custom = custom_lexicon_path or (env_custom if env_custom else None)
        if effective_custom is not None:
            self.load_custom_overlay(effective_custom)

        self._anti_confusion_ac: TokenAhoCorasick[AntiConfusionRule] | None = None
        self._ortho_phrase_ac: TokenAhoCorasick[str] | None = None
        self._rebuild_automata()

    def _add_anti_confusion_rule(self, raw_rule: dict[str, Any]) -> None:
        rule = AntiConfusionRule(
            non_saudi=str(raw_rule["non_saudi"]).strip(),
            origin=str(raw_rule.get("origin", "other")).strip(),
            saudi_white=str(raw_rule["saudi_white"]).strip(),
            alt=tuple(str(x).strip() for x in raw_rule.get("alt", [])),
            note=str(raw_rule.get("note", "")).strip(),
        )
        key = light_arabic_key(rule.non_saudi)
        self._anti_confusion_index[key] = rule
        self._anti_confusion.append(rule)

    def _rebuild_automata(self) -> None:
        ac_drift: TokenAhoCorasick[AntiConfusionRule] = TokenAhoCorasick(light_normalize=True)
        for rule in self._anti_confusion:
            ac_drift.add_phrase(rule.non_saudi, rule)
        ac_drift.compile()
        self._anti_confusion_ac = ac_drift

        ac_ortho: TokenAhoCorasick[str] = TokenAhoCorasick(light_normalize=False)
        # Canonical multi-word orthographic normalizations
        builtin_multi_ortho = (
            ("ان شاء الله", "إن شاء الله"),
            ("انشاءالله", "إن شاء الله"),
            ("إنشالله", "إن شاء الله"),
            ("انشالله", "إن شاء الله"),
            ("إن شاءالله", "إن شاء الله"),
            ("ماشاء الله", "ما شاء الله"),
            ("ماشاءالله", "ما شاء الله"),
            ("مشاءالله", "ما شاء الله"),
            ("ماشالله", "ما شاء الله"),
            ("جزاك الله خيرا", "جزاك الله خير"),
            ("لا هنت", "لاهنت"),
            ("وش لونك", "وشلونك"),
            ("وش لون", "وشلون"),
            ("كل شيء", "كل شي"),
            ("كلشئ", "كل شي"),
            ("وكل شيء", "وكل شي"),
            ("وكلشئ", "وكل شي"),
            ("ولا شيء", "ولا شي"),
            ("أول شيء", "أول شي"),
            ("أهم شيء", "أهم شي"),
            ("في المائة", "بالمية"),
            ("في المئة", "بالمية"),
        )
        for src, dst in builtin_multi_ortho:
            ac_ortho.add_phrase(src, dst)
        for src, dst in self._variant_to_canonical.items():
            if " " in src:
                ac_ortho.add_phrase(src, dst)
        ac_ortho.compile()
        self._ortho_phrase_ac = ac_ortho

    def apply_pack(self, pack: DialectPack, *, override_existing: bool = False) -> int:
        """Merge a modular DialectPack into this lexicon instance."""
        if pack.id not in self._loaded_pack_ids:
            self._loaded_pack_ids.append(pack.id)
        added = 0
        for item in pack.entries:
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            if term in self._seen_terms and not override_existing:
                continue
            entry = self._register_raw_entry(item, pack_id=pack.id, is_custom=override_existing)
            if override_existing:
                self._index_entry(entry, is_custom=True)
            added += 1

        for src, dst in pack.orthographic_map:
            self._variant_to_canonical[src] = dst
            added += 1

        for rule_dict in pack.anti_confusion_map:
            self._add_anti_confusion_rule(rule_dict)

        for hall in pack.hallucination_blacklist:
            if hall not in self._hallucination_blacklist:
                self._hallucination_blacklist.append(hall)

        self._anti_confusion_ac = None
        self._ortho_phrase_ac = None
        return added

    def _register_raw_entry(
        self,
        item: dict[str, Any],
        *,
        pack_id: str = "core",
        is_custom: bool = False,
    ) -> LexiconEntry:
        entry = LexiconEntry(
            id=str(item.get("id", f"{pack_id}_{len(self._entries) + 1:04d}")),
            term=str(item["term"]).strip(),
            vocalized=str(item.get("vocalized") or item["term"]).strip(),
            ipa_hint=str(item.get("ipa_hint", "")).strip(),
            category=str(item.get("category", "custom")).strip(),
            pos=str(item.get("pos", "phrase")).strip(),
            register=str(item.get("register", "universal")).strip(),
            meaning_ar=str(item.get("meaning_ar", "")).strip(),
            meaning_en=str(item.get("meaning_en", "")).strip(),
            variants=tuple(str(v).strip() for v in item.get("variants", []) if str(v).strip()),
            example=str(item.get("example", "")).strip(),
            pack_id=pack_id,
        )
        self._seen_terms.add(entry.term)
        self._entries.append(entry)
        self._index_entry(entry, is_custom=is_custom)
        return entry

    def _index_entry(self, entry: LexiconEntry, *, is_custom: bool = False) -> None:
        plain_term = strip_diacritics(entry.term)
        light_term = light_arabic_key(entry.term)

        self._exact_map[entry.term] = entry
        self._exact_map[plain_term] = entry
        self._light_map.setdefault(light_term, entry)
        self._vocalize_map[entry.term] = entry.vocalized
        self._vocalize_map[plain_term] = entry.vocalized

        if plain_term != entry.term and " " not in entry.term:
            self._variant_to_canonical.setdefault(plain_term, entry.term)

        phonetic_surfaces = [entry.term]
        for variant in entry.variants:
            plain_var = strip_diacritics(variant)
            light_var = light_arabic_key(variant)
            self._exact_map.setdefault(variant, entry)
            self._exact_map.setdefault(plain_var, entry)
            self._light_map.setdefault(light_var, entry)

            if is_custom or _is_pure_orthographic_variant(variant, entry.term):
                self._vocalize_map.setdefault(variant, entry.vocalized)
                self._vocalize_map.setdefault(plain_var, entry.vocalized)
                phonetic_surfaces.append(variant)
                if variant != entry.term:
                    self._variant_to_canonical[variant] = entry.term
                    self._variant_to_canonical[plain_var] = entry.term

        for surface in phonetic_surfaces:
            parts = tuple(strip_diacritics(surface).split())
            if len(parts) > 1:
                self._phrase_trie.insert(parts, entry, overwrite=is_custom)
                bucket = self._multiword_by_first.setdefault(parts[0], [])
                bucket.append((parts, entry))
                bucket.sort(key=lambda pair: len(pair[0]), reverse=True)

    def load_custom_overlay(self, path: str | os.PathLike[str]) -> int:
        """Load an external user lexicon file at runtime without mutating bundled data."""
        resolved = Path(os.path.expanduser(str(path)))
        if not resolved.is_file():
            raise FileNotFoundError(f"Custom lexicon file not found: {resolved}")

        with resolved.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        added = 0
        if isinstance(payload, dict) and ("entries" in payload or "orthographic_map" in payload):
            for item in payload.get("entries", []):
                entry = self._register_raw_entry(item, pack_id="custom_overlay", is_custom=True)
                self._index_entry(entry, is_custom=True)
                added += 1
            for src, dst in payload.get("orthographic_map", {}).items():
                self._variant_to_canonical[str(src).strip()] = str(dst).strip()
                added += 1
            for hall in payload.get("hallucination_blacklist", []):
                if hall not in self._hallucination_blacklist:
                    self._hallucination_blacklist.append(str(hall))
        elif isinstance(payload, dict):
            for key, val in payload.items():
                if isinstance(val, dict):
                    item = {"term": key, **val}
                    entry = self._register_raw_entry(item, pack_id="custom_overlay", is_custom=True)
                    self._index_entry(entry, is_custom=True)
                else:
                    self._variant_to_canonical[str(key).strip()] = str(val).strip()
                added += 1
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "term" in item:
                    entry = self._register_raw_entry(item, pack_id="custom_overlay", is_custom=True)
                    self._index_entry(entry, is_custom=True)
                    added += 1
        self._rebuild_automata()
        return added

    @property
    def entries(self) -> tuple[LexiconEntry, ...]:
        return tuple(self._entries)

    @property
    def loaded_pack_ids(self) -> tuple[str, ...]:
        return tuple(self._loaded_pack_ids)

    @property
    def phrase_trie(self) -> PhraseRadixTrie[LexiconEntry]:
        return self._phrase_trie

    @property
    def anti_confusion_automaton(self) -> TokenAhoCorasick[AntiConfusionRule]:
        if self._anti_confusion_ac is None:
            self._rebuild_automata()
        assert self._anti_confusion_ac is not None
        return self._anti_confusion_ac

    @property
    def ortho_phrase_automaton(self) -> TokenAhoCorasick[str]:
        if self._ortho_phrase_ac is None:
            self._rebuild_automata()
        assert self._ortho_phrase_ac is not None
        return self._ortho_phrase_ac

    @property
    def anti_confusion_rules(self) -> tuple[AntiConfusionRule, ...]:
        return tuple(self._anti_confusion)

    @property
    def hallucination_blacklist(self) -> tuple[str, ...]:
        return tuple(self._hallucination_blacklist)

    @property
    def multiword_index(self) -> dict[str, list[tuple[tuple[str, ...], LexiconEntry]]]:
        return self._multiword_by_first

    @property
    def variant_map(self) -> dict[str, str]:
        return self._variant_to_canonical

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, query: str) -> LexiconEntry | None:
        """O(1) lookup by exact form, unvocalized form, variant, or light-normalized key."""
        q = query.strip()
        if not q:
            return None
        hit = self._exact_map.get(q)
        if hit is not None:
            return hit
        plain = strip_diacritics(q)
        hit = self._exact_map.get(plain)
        if hit is not None:
            return hit
        return self._light_map.get(light_arabic_key(plain))

    def lookup_anti_confusion(self, phrase: str) -> AntiConfusionRule | None:
        """Check if a word/phrase is a known non-Saudi or stiff-MSA false friend."""
        return self._anti_confusion_index.get(light_arabic_key(phrase))

    def normalize_token(self, token: str) -> str:
        """Return canonical White Saudi spelling if token is a registered variant."""
        direct = self._variant_to_canonical.get(token)
        if direct is not None:
            return direct
        plain = strip_diacritics(token)
        return self._variant_to_canonical.get(plain, token)

    def vocalize_token(self, token: str) -> str | None:
        """Return vocalized phonetic form if token (or pure spelling variant) exists in the lexicon."""
        voc = self._vocalize_map.get(token)
        if voc is not None:
            return voc
        return self._vocalize_map.get(strip_diacritics(token))

    def search(
        self,
        query: str = "",
        *,
        category: str | None = None,
        register: str | None = None,
        pack_id: str | None = None,
        limit: int = 20,
    ) -> list[LexiconEntry]:
        """Search entries by Arabic/English substring, category, register, or pack_id."""
        q = query.strip()
        q_light = light_arabic_key(q) if q else ""
        q_lower = q.lower()

        results: list[LexiconEntry] = []
        exact = self.lookup(q) if q else None
        if exact is not None:
            if (
                (category is None or exact.category == category)
                and (register is None or exact.register == register)
                and (pack_id is None or exact.pack_id == pack_id)
            ):
                results.append(exact)

        for entry in self._entries:
            if exact is not None and entry.id == exact.id:
                continue
            if category is not None and entry.category != category:
                continue
            if register is not None and entry.register != register:
                continue
            if pack_id is not None and entry.pack_id != pack_id:
                continue
            if not q:
                results.append(entry)
            else:
                haystack_ar = f"{light_arabic_key(entry.term)} {' '.join(light_arabic_key(v) for v in entry.variants)} {light_arabic_key(entry.meaning_ar)}"
                haystack_en = f"{entry.meaning_en.lower()} {entry.ipa_hint.lower()}"
                if q_light in haystack_ar or q_lower in haystack_en:
                    results.append(entry)
            if len(results) >= limit:
                break
        return results

    def detect_dialect_drift(self, text: str) -> list[dict[str, Any]]:
        """Detect non-Saudi dialectisms or stiff MSA phrases in single-pass O(N) time."""
        matches = self.anti_confusion_automaton.select_non_overlapping(text)
        seen: set[str] = set()
        findings: list[dict[str, Any]] = []
        for m in matches:
            rule = m.value
            if rule.non_saudi not in seen:
                seen.add(rule.non_saudi)
                item = rule.to_dict()
                item["span"] = [m.start_char, m.end_char]
                item["matched_text"] = m.matched_text
                findings.append(item)
        return findings

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.category] = counts.get(entry.category, 0) + 1
        return counts

    def pack_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.pack_id] = counts.get(entry.pack_id, 0) + 1
        return counts

    def stats(self) -> dict[str, Any]:
        return {
            "entry_count": len(self._entries),
            "variant_count": len(self._variant_to_canonical),
            "phrase_trie_size": len(self._phrase_trie),
            "anti_confusion_count": len(self._anti_confusion),
            "hallucination_patterns": len(self._hallucination_blacklist),
            "loaded_packs": list(self._loaded_pack_ids),
            "pack_counts": self.pack_counts(),
            "categories": self.categories(),
        }


@lru_cache(maxsize=8)
def get_default_lexicon(custom_lexicon_path: str | None = None) -> NajdiLexicon:
    """Return a cached NajdiLexicon singleton (with all active packs) for <0.04ms hot-path calls."""
    return NajdiLexicon(custom_lexicon_path=custom_lexicon_path)


def iter_terms(lexicon: NajdiLexicon | None = None) -> Iterable[str]:
    lex = lexicon or get_default_lexicon()
    for entry in lex.entries:
        yield entry.term
