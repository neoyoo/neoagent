from __future__ import annotations
import re
from pathlib import Path

_INDEX_FILE = "MEMORY.md"


class MemoryStore:
    """File-based memory storage.

    Layout:
        memory_dir/
            MEMORY.md          ← index: markdown link list
            {topic}.md         ← per-topic content files
    """

    def __init__(self, memory_dir: Path) -> None:
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def index_path(self) -> Path:
        return self._dir / _INDEX_FILE

    def _safe_path(self, filename: str) -> Path:
        """Resolve filename and verify it's within the memory directory."""
        path = (self._dir / filename).resolve()
        if not path.is_relative_to(self._dir.resolve()):
            raise ValueError(f"Memory filename {filename!r} escapes memory directory")
        return path

    def read_index(self) -> str:
        if not self.index_path.exists():
            return ""
        return self.index_path.read_text(encoding="utf-8")

    def write_index(self, content: str) -> None:
        self.index_path.write_text(content, encoding="utf-8")

    def read_topic(self, filename: str) -> str:
        path = self._safe_path(filename)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_topic(self, filename: str, content: str) -> None:
        path = self._safe_path(filename)
        path.write_text(content, encoding="utf-8")

    def delete_topic(self, filename: str) -> None:
        path = self._safe_path(filename)
        if path.exists():
            path.unlink()

    def list_topics(self) -> list[tuple[str, str]]:
        """Return [(filename, description)] from MEMORY.md index.

        Index format: each topic line is '- [description](filename)'
        """
        index = self.read_index()
        topics: list[tuple[str, str]] = []
        for line in index.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"-\s*\[([^\]]+)\]\(([^)]+)\)", line)
            if m:
                description, filename = m.group(1), m.group(2)
                topics.append((filename, description))
        return topics
