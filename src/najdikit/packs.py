"""Compiled Aho-Corasick / Radix Trie engines and Modular Domain Pack Registry (v2.0)."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Iterable, Sequence, TypeVar

T = TypeVar("T")

_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_ALEF_RE = re.compile(r"[أإآٱ]")
_WORD_TOKEN_RE = re.compile(r"[\u0621-\u064A\u0671\u064B-\u065F\u0670]+|[A-Za-z0-9]+")
_BUILTIN_PACKS_DIR = Path(__file__).resolve().parent / "data" / "packs"


def _norm_token(tok: str, *, light: bool = False) -> str:
    plain = _DIACRITICS_RE.sub("", tok)
    return _ALEF_RE.sub("ا", plain) if light else plain


@dataclass(frozen=True, slots=True)
class AutomatonMatch(Generic[T]):
    """A matched token span produced by TokenAhoCorasick."""

    start_token: int
    end_token: int  # exclusive
    start_char: int
    end_char: int
    matched_text: str
    value: T


class TokenAhoCorasick(Generic[T]):
    """O(N) multi-pattern Aho-Corasick automaton over word tokens.

    Scales to 10,000+ single-word and multi-word Arabic phrases in a single linear pass
    without regex backtracking overhead.
    """

    __slots__ = ("_goto", "_fail", "_output", "_light", "_compiled", "_pattern_count")

    def __init__(self, *, light_normalize: bool = False) -> None:
        self._goto: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._output: list[list[tuple[int, T]]] = [[]]
        self._light = light_normalize
        self._compiled = False
        self._pattern_count = 0

    @property
    def pattern_count(self) -> int:
        return self._pattern_count

    def add_phrase(self, phrase: str, value: T) -> None:
        tokens = [
            _norm_token(m.group(0), light=self._light)
            for m in _WORD_TOKEN_RE.finditer(phrase)
        ]
        if not tokens:
            return
        state = 0
        goto = self._goto
        for tok in tokens:
            nxt = goto[state].get(tok)
            if nxt is None:
                nxt = len(goto)
                goto[state][tok] = nxt
                goto.append({})
                self._fail.append(0)
                self._output.append([])
            state = nxt
        # Keep longest pattern first at each state
        self._output[state].append((len(tokens), value))
        self._pattern_count += 1
        self._compiled = False

    def compile(self) -> "TokenAhoCorasick[T]":
        if self._compiled:
            return self
        queue: deque[int] = deque()
        goto = self._goto
        fail = self._fail
        output = self._output

        for _tok, nxt in goto[0].items():
            fail[nxt] = 0
            queue.append(nxt)

        while queue:
            r = queue.popleft()
            for tok, u in goto[r].items():
                queue.append(u)
                v = fail[r]
                while v != 0 and tok not in goto[v]:
                    v = fail[v]
                fail[u] = goto[v].get(tok, 0)
                if output[fail[u]]:
                    merged = output[u] + output[fail[u]]
                    merged.sort(key=lambda pair: pair[0], reverse=True)
                    output[u] = merged

        self._compiled = True
        return self

    def scan(self, text: str) -> list[AutomatonMatch[T]]:
        """Return all multi-pattern matches in `text` in a single O(N) pass."""
        if not text or self._pattern_count == 0:
            return []
        if not self._compiled:
            self.compile()

        spans: list[tuple[str, int, int]] = [
            (_norm_token(m.group(0), light=self._light), m.start(), m.end())
            for m in _WORD_TOKEN_RE.finditer(text)
        ]
        if not spans:
            return []

        goto = self._goto
        fail = self._fail
        output = self._output
        state = 0
        matches: list[AutomatonMatch[T]] = []

        for idx, (tok, _c_start, c_end) in enumerate(spans):
            while state != 0 and tok not in goto[state]:
                state = fail[state]
            state = goto[state].get(tok, 0)
            outs = output[state]
            if outs:
                for length, val in outs:
                    start_tok = idx - length + 1
                    s_char = spans[start_tok][1]
                    matches.append(
                        AutomatonMatch(
                            start_token=start_tok,
                            end_token=idx + 1,
                            start_char=s_char,
                            end_char=c_end,
                            matched_text=text[s_char:c_end],
                            value=val,
                        )
                    )
        return matches

    def select_non_overlapping(self, text: str) -> list[AutomatonMatch[T]]:
        """Select longest-leftmost non-overlapping matches in O(N + M log M)."""
        all_matches = self.scan(text)
        if not all_matches:
            return []
        all_matches.sort(key=lambda m: (m.start_token, -(m.end_token - m.start_token)))
        selected: list[AutomatonMatch[T]] = []
        cursor = -1
        for m in all_matches:
            if m.start_token >= cursor:
                selected.append(m)
                cursor = m.end_token
        return selected

    def replace_all(self, text: str) -> tuple[str, int]:
        """Single-pass non-overlapping phrase replacement when `T` is `str`."""
        chosen = self.select_non_overlapping(text)
        if not chosen:
            return text, 0
        chunks: list[str] = []
        last_pos = 0
        replaced = 0
        for m in chosen:
            replacement = str(m.value)
            chunks.append(text[last_pos : m.start_char])
            chunks.append(replacement)
            last_pos = m.end_char
            if replacement != m.matched_text:
                replaced += 1
        chunks.append(text[last_pos:])
        return "".join(chunks), replaced


class PhraseRadixTrie(Generic[T]):
    """Compressed Radix Trie over normalized word tuples for O(L) greedy longest-prefix lookup."""

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        # Node structure: [children_dict, terminal_value_or_None]
        self._root: list[Any] = [{}, None]
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, words: Sequence[str], value: T, *, overwrite: bool = True) -> None:
        if not words:
            return
        node = self._root
        for w in words:
            children: dict[str, list[Any]] = node[0]
            nxt = children.get(w)
            if nxt is None:
                nxt = [{}, None]
                children[w] = nxt
            node = nxt
        if node[1] is None:
            self._size += 1
            node[1] = value
        elif overwrite:
            node[1] = value

    def match_longest(self, words: Sequence[str], start: int = 0) -> tuple[int, T] | None:
        """Return `(matched_word_len, value)` for the longest phrase starting at `words[start]`."""
        node = self._root
        best: tuple[int, T] | None = None
        n = len(words)
        idx = start
        while idx < n:
            nxt = node[0].get(words[idx])
            if nxt is None:
                break
            node = nxt
            idx += 1
            if node[1] is not None:
                best = (idx - start, node[1])
        return best


@dataclass(frozen=True, slots=True)
class DialectPack:
    """Modular, immutable domain or regional dialect pack."""

    id: str
    name: str
    version: str = "2.0.0"
    domain: str = "general"
    priority: int = 50
    description: str = ""
    entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    orthographic_map: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    anti_confusion_map: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    hallucination_blacklist: tuple[str, ...] = field(default_factory=tuple)
    source_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str | None = None) -> "DialectPack":
        pack_id = str(data.get("id") or "custom_pack").strip()
        ortho_raw = data.get("orthographic_map") or {}
        ortho_pairs = tuple((str(k).strip(), str(v).strip()) for k, v in ortho_raw.items())
        return cls(
            id=pack_id,
            name=str(data.get("name") or pack_id).strip(),
            version=str(data.get("version") or "2.0.0").strip(),
            domain=str(data.get("domain") or "general").strip(),
            priority=int(data.get("priority", 50)),
            description=str(data.get("description") or "").strip(),
            entries=tuple(dict(item) for item in data.get("entries", [])),
            orthographic_map=ortho_pairs,
            anti_confusion_map=tuple(dict(item) for item in data.get("anti_confusion_map", [])),
            hallucination_blacklist=tuple(str(x) for x in data.get("hallucination_blacklist", [])),
            source_path=source_path,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "priority": self.priority,
            "entry_count": len(self.entries),
            "description": self.description,
        }


class DialectPackRegistry:
    """Thread-safe registry managing built-in and external DialectPacks with immutable snapshots."""

    def __init__(self, *, load_builtins: bool = True, load_env_dir: bool = True) -> None:
        self._lock = threading.RLock()
        self._packs: dict[str, DialectPack] = {}
        self._snapshot: tuple[DialectPack, ...] = ()
        self._revision: int = 0

        if load_builtins and _BUILTIN_PACKS_DIR.is_dir():
            self.load_packs_dir(_BUILTIN_PACKS_DIR)

        if load_env_dir:
            env_dir = os.environ.get("NAJDIKIT_PACKS_DIR", "").strip()
            if env_dir:
                resolved = Path(os.path.expanduser(env_dir))
                if resolved.is_dir():
                    self.load_packs_dir(resolved)

    def _rebuild_snapshot_unlocked(self) -> None:
        # Lower priority number loads first; higher priority number overrides on custom collisions
        ordered = sorted(self._packs.values(), key=lambda p: (p.priority, p.id))
        self._snapshot = tuple(ordered)
        self._revision += 1

    @property
    def revision(self) -> int:
        return self._revision

    def snapshot(self) -> tuple[DialectPack, ...]:
        """Return an immutable ordered tuple of active packs (lock-free read)."""
        return self._snapshot

    def register_pack(self, pack: DialectPack) -> DialectPack:
        with self._lock:
            self._packs[pack.id] = pack
            self._rebuild_snapshot_unlocked()
        return pack

    def unregister_pack(self, pack_id: str) -> bool:
        with self._lock:
            removed = self._packs.pop(pack_id, None) is not None
            if removed:
                self._rebuild_snapshot_unlocked()
            return removed

    def get_pack(self, pack_id: str) -> DialectPack | None:
        return self._packs.get(pack_id)

    def load_pack_file(self, path: str | os.PathLike[str]) -> DialectPack:
        resolved = Path(os.path.expanduser(str(path)))
        if not resolved.is_file():
            raise FileNotFoundError(f"DialectPack file not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if "id" not in raw:
            raw = {"id": resolved.stem, **raw}
        pack = DialectPack.from_dict(raw, source_path=resolved.name)
        return self.register_pack(pack)

    def load_packs_dir(self, directory: str | os.PathLike[str]) -> list[DialectPack]:
        resolved = Path(os.path.expanduser(str(directory)))
        if not resolved.is_dir():
            raise NotADirectoryError(f"DialectPack directory not found: {resolved}")
        loaded: list[DialectPack] = []
        for json_path in sorted(resolved.glob("*.json")):
            loaded.append(self.load_pack_file(json_path))
        return loaded

    def list_packs(self) -> list[dict[str, Any]]:
        return [p.summary() for p in self._snapshot]


_DEFAULT_REGISTRY: DialectPackRegistry | None = None
_REGISTRY_INIT_LOCK = threading.Lock()


def get_default_pack_registry() -> DialectPackRegistry:
    """Return the process-wide default DialectPackRegistry singleton."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        with _REGISTRY_INIT_LOCK:
            if _DEFAULT_REGISTRY is None:
                _DEFAULT_REGISTRY = DialectPackRegistry()
    return _DEFAULT_REGISTRY
