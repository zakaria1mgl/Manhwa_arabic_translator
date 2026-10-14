from __future__ import annotations

import json
import logging
import os
import threading
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class Glossary:
    """Persistent EN->AR terminology memory (names, places, skills, SFX)."""

    def __init__(self, path: str, *, auto_learn: bool = True):
        self.path = path
        self.auto_learn = auto_learn
        self.terms: Dict[str, str] = {}
        self.display: Dict[str, str] = {}  # normalized key -> original casing
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.terms = {str(k): str(v) for k, v in data.items()}
            except Exception as e:
                log.warning("glossary load failed: %s", e)

    def save(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.terms, f, ensure_ascii=False, indent=2)

    def get(self, en: str) -> Optional[str]:
        return self.terms.get(_norm_key(en))

    def set(self, en: str, ar: str) -> None:
        key = _norm_key(en)
        self.terms[key] = ar
        self.display[key] = en.strip()
        if self.auto_learn:
            self.save()

    def delete(self, en: str) -> None:
        self.terms.pop(_norm_key(en), None)
        self.save()

    def as_prompt_block(self, max_entries: int = 120) -> str:
        if not self.terms:
            return ""
        items = sorted(self.terms.items())
        lines = [f"- {self.display.get(k, k)} => {v}" for k, v in items[:max_entries]]
        return "Glossary (MUST be followed exactly):\n" + "\n".join(lines)

    def to_rows(self) -> List[List[str]]:
        return [[self.display.get(k, k), v] for k, v in sorted(self.terms.items())]

    def replace_all(self, rows: List[List[str]]) -> None:
        self.terms = {}
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                self.terms[_norm_key(row[0].strip())] = row[1].strip()
        self.save()


def _norm_key(s: str) -> str:
    return " ".join(s.strip().lower().split())
