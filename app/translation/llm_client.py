from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..config import AppConfig, LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Minimal OpenAI-compatible chat client (works with Gemini's
    OpenAI-compatibility endpoint and any compatible server).

    When constructed with an AppConfig the LLM settings are read LAZILY on
    every call, so changes applied in the Settings tab take effect without
    recreating the pipeline. A plain LLMConfig snapshot also works (tests).
    """

    def __init__(self, cfg):
        if isinstance(cfg, LLMConfig):
            self._static: Optional[LLMConfig] = cfg
            self._app: Optional["AppConfig"] = None
        else:
            self._static = None
            self._app = cfg

    @property
    def cfg(self) -> LLMConfig:
        if self._static is not None:
            return self._static
        assert self._app is not None
        return self._app.llm

    @property
    def ready(self) -> bool:
        c = self.cfg
        return bool(c.api_key and c.base_url and c.model)

    def chat(self, messages: List[Dict[str, str]], *,
             temperature: Optional[float] = None,
             json_mode: bool = False) -> str:
        assert self.ready, "LLM client not configured"
        import requests

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body,
                                     timeout=self.cfg.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                log.warning("LLM call failed (attempt %d/%d): %s; retry in %ss",
                            attempt + 1, self.cfg.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


def extract_json(text: str) -> Any:
    """Robustly extract the first JSON object/array from an LLM reply."""
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # find outermost braces / brackets
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        # try progressively trimming trailing junk after first candidate
        m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError("no valid JSON found in LLM reply")
