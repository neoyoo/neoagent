from __future__ import annotations
import logging
from neoagent.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FILES = 5
_DEFAULT_MAX_CONTENT_CHARS = 2000


class MemoryRetriever:
    """Lexical retrieval from MemoryStore.

    Scores topic descriptions by keyword overlap with query.
    Returns index + top-N matching topic file contents.
    """

    def __init__(
        self,
        store: MemoryStore,
        max_files: int = _DEFAULT_MAX_FILES,
        max_content_chars: int = _DEFAULT_MAX_CONTENT_CHARS,
    ) -> None:
        self._store = store
        self._max_files = max_files
        self._max_content_chars = max_content_chars

    def retrieve(self, query: str | None = None) -> str:
        """Return formatted memory content.

        If query is None: return index only (no topic file content).
        If query given: return index + top-N matched topic files.
        """
        index = self._store.read_index()
        if not index:
            return ""

        if query is None:
            return f"# Memory\n{index}"

        topics = self._store.list_topics()
        if not topics:
            return f"# Memory\n{index}"

        query_words = set(query.lower().split())
        scored = [
            (sum(1 for w in query_words if w in desc.lower()), fname, desc)
            for fname, desc in topics
        ]
        scored.sort(key=lambda x: -x[0])

        top = [x for x in scored if x[0] > 0] or scored
        top = top[: self._max_files]

        parts = [f"# Memory\n{index}"]
        for _, filename, description in top:
            try:
                content = self._store.read_topic(filename)
            except ValueError:
                logger.warning("Skipping invalid memory topic: %s", filename)
                continue
            if content:
                parts.append(f"## {description}\n{content[: self._max_content_chars]}")

        return "\n\n".join(parts)
