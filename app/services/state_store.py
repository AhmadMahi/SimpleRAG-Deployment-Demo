from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(
        self,
        file_path: str,
        cache_size: int,
        conversation_history_size: int,
    ) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache_size = cache_size
        self.conversation_history_size = conversation_history_size

        self._lock = Lock()
        self._documents: dict[str, dict[str, Any]] = {}
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._conversations: list[dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._flush_nolock()
            return

        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._flush_nolock()
            return

        self._documents = raw.get("documents", {})

        self._cache = OrderedDict()
        for item in raw.get("cache_entries", []):
            key = item.get("key")
            value = item.get("value")
            if isinstance(key, str) and isinstance(value, dict):
                self._cache[key] = value

        self._conversations = raw.get("conversations", [])

    def _flush_nolock(self) -> None:
        serializable = {
            "documents": self._documents,
            "cache_entries": [
                {"key": key, "value": value} for key, value in self._cache.items()
            ],
            "conversations": self._conversations,
        }
        self.file_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def add_document(self, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._documents[metadata["id"]] = deepcopy(metadata)
            self._flush_nolock()

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            docs = list(self._documents.values())

        docs.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
        return docs

    def get_document_ids(self) -> set[str]:
        with self._lock:
            return set(self._documents.keys())

    def get_cached_answer(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._cache.get(key)
            if payload is None:
                return None

            self._cache.move_to_end(key)
            self._flush_nolock()
            return deepcopy(payload)

    def set_cached_answer(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            payload_to_store = deepcopy(payload)
            payload_to_store["cached_at"] = utc_now_iso()

            self._cache[key] = payload_to_store
            self._cache.move_to_end(key)

            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

            self._flush_nolock()

    def clear_cache(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._flush_nolock()
            return count

    def cache_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def add_conversation(self, item: dict[str, Any]) -> None:
        with self._lock:
            entry = deepcopy(item)
            if "asked_at" not in entry:
                entry["asked_at"] = utc_now_iso()

            self._conversations.append(entry)
            if len(self._conversations) > self.conversation_history_size:
                self._conversations = self._conversations[
                    -self.conversation_history_size :
                ]
            self._flush_nolock()

    def list_conversations(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            recent = self._conversations[-limit:]
            return list(reversed(deepcopy(recent)))
