# src/Model/memory.py
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

from llm_config import LLMConfig


@dataclass
class UserMemory:
    cfg: LLMConfig
    _store: Dict[str, Any] = field(default_factory=dict)
    _loaded: bool = False

    # ---------- Storage ----------

    def _load(self) -> None:
        if self._loaded:
            return
        path = self.cfg.memory_file
        if path.exists():
            try:
                self._store = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._store = {}
        self._loaded = True

    def _save(self) -> None:
        path = self.cfg.memory_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Profile ----------

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        self._load()
        return self._store.setdefault(user_id, {}).setdefault("profile", {})

    def update_profile(self, user_id: str, **fields) -> None:
        self._load()
        profile = self._store.setdefault(user_id, {}).setdefault("profile", {})
        for k, v in fields.items():
            profile[k] = v
        self._save()

    # ---------- Conversation history ----------

    def add_turn(self, user_id: str, user_msg: str, assistant_msg: str) -> None:
        self._load()
        conv = self._store.setdefault(user_id, {}).setdefault("history", [])
        conv.append({"user": user_msg, "assistant": assistant_msg})
        # keep last 20 turns to avoid infinite growth
        if len(conv) > 20:
            del conv[:-20]
        self._save()

    def get_recent_history(self, user_id: str, limit: int = 6) -> List[Dict[str, str]]:
        self._load()
        conv = self._store.get(user_id, {}).get("history", [])
        return conv[-limit:]
