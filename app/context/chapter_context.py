from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..glossary.glossary import Glossary
from ..utils.types import PageResult, RegionType, TextRegion

log = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    name: str
    gender: str = "unknown"
    personality: str = ""
    speaking_style: str = ""
    formality: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)
    preferred_terms: Dict[str, str] = field(default_factory=dict)
    sample_lines: List[str] = field(default_factory=list)
    first_seen_page: int = 0
    last_seen_page: int = 0


class ChapterContext:
    """Persistent chapter/session context: characters, story, terminology."""

    def __init__(self, chapter_id: str = "default", persist_dir: str = "."):
        self.chapter_id = chapter_id
        self.persist_path = os.path.join(persist_dir, f"context_{chapter_id}.json")
        self.characters: Dict[str, CharacterProfile] = {}
        self.story_summary: str = ""
        self.page_summaries: List[Dict[str, Any]] = []
        self.recent_dialogue: List[Dict[str, str]] = []  # last N lines
        self.sfx_choices: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.load()

    # persistence ------------------------------------------------------------
    def load(self) -> None:
        if os.path.isfile(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.story_summary = data.get("story_summary", "")
                self.page_summaries = data.get("page_summaries", [])
                self.recent_dialogue = data.get("recent_dialogue", [])[-30:]
                self.sfx_choices = data.get("sfx_choices", {})
                for name, p in data.get("characters", {}).items():
                    self.characters[name] = CharacterProfile(
                        **{k: v for k, v in p.items()
                           if k in CharacterProfile.__dataclass_fields__})
            except Exception as e:
                log.warning("context load failed: %s", e)

    def save(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            data = {
                "story_summary": self.story_summary,
                "page_summaries": self.page_summaries[-50:],
                "recent_dialogue": self.recent_dialogue[-30:],
                "sfx_choices": self.sfx_choices,
                "characters": {k: asdict(v) for k, v in self.characters.items()},
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)

    # accessors ----------------------------------------------------------------
    def remember_page(self, page_number: int, result: PageResult) -> None:
        self.page_summaries.append({
            "page": page_number,
            "summary": result.page_summary,
            "dialogue": [
                {"speaker": r.speaker or "?", "en": r.english, "ar": r.arabic}
                for r in result.regions if r.english
            ],
        })
        for d in result.regions:
            if d.english and d.region_type in (RegionType.DIALOGUE, RegionType.THOUGHT):
                self.recent_dialogue.append({
                    "speaker": d.speaker or "?", "en": d.english, "ar": d.arabic,
                })
        self.recent_dialogue = self.recent_dialogue[-30:]

    def upsert_character(self, profile: CharacterProfile) -> None:
        cur = self.characters.get(profile.name)
        if cur is None:
            self.characters[profile.name] = profile
        else:
            for k, v in asdict(profile).items():
                if v and v != "unknown":
                    setattr(cur, k, v)
            cur.last_seen_page = profile.last_seen_page

    def character_block(self) -> str:
        if not self.characters:
            return ""
        lines = []
        for name, c in self.characters.items():
            seg = f"- {name} ({c.gender})"
            if c.personality:
                seg += f" | personality: {c.personality}"
            if c.speaking_style:
                seg += f" | style: {c.speaking_style}"
            if c.formality:
                seg += f" | formality: {c.formality}"
            lines.append(seg)
        return "Known characters:\n" + "\n".join(lines)

    def recent_dialogue_block(self, n: int = 12) -> str:
        if not self.recent_dialogue:
            return ""
        lines = [f"{d['speaker']}: {d['en']}" for d in self.recent_dialogue[-n:]]
        return "Recent dialogue (earlier pages, EN):\n" + "\n".join(lines)

    def reset(self) -> None:
        self.characters.clear()
        self.story_summary = ""
        self.page_summaries.clear()
        self.recent_dialogue.clear()
        self.sfx_choices.clear()
