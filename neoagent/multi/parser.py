from __future__ import annotations

from pathlib import Path

import yaml

from neoagent.multi.worker import WorkerCard


def parse_worker_md(content: str) -> WorkerCard:
    """Parse a .md file with YAML frontmatter into a WorkerCard.

    The file must begin with a ``---`` marker followed by YAML frontmatter
    and a closing ``---`` marker.  Everything after the closing marker is
    treated as the worker's instruction (system prompt body).

    Raises:
        ValueError: If frontmatter is missing, YAML is malformed, or the
                    required ``name`` field is absent.
    """
    if not content.startswith("---"):
        raise ValueError("No frontmatter found: file must start with '---'")

    # Split off the frontmatter block.  We expect at least two '---' markers.
    parts = content.split("---", maxsplit=2)
    # parts[0] is the empty string before the first '---'
    # parts[1] is the YAML content
    # parts[2] is the body (may be absent if there is no closing marker)
    if len(parts) < 3:
        raise ValueError("No frontmatter found: missing closing '---' marker")

    raw_yaml = parts[1]
    body = parts[2]

    try:
        frontmatter = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in frontmatter: {exc}") from exc

    if not isinstance(frontmatter, dict):
        raise ValueError("Malformed YAML frontmatter: expected a mapping")

    name: str | None = frontmatter.get("name")
    if not name:
        raise ValueError("Missing required field 'name' in frontmatter")

    description: str = frontmatter.get("description") or ""
    model: str | None = frontmatter.get("model") or None

    raw_tags = frontmatter.get("tags") or []
    tags: tuple[str, ...] = tuple(str(t) for t in raw_tags)

    raw_tools = frontmatter.get("tools") or []
    tools: tuple[str, ...] = tuple(str(t) for t in raw_tools)

    return WorkerCard(
        name=name,
        description=description,
        instruction=body,
        tags=tags,
        model=model,
        tools=tools,
    )


def load_workers(directory: Path) -> list[WorkerCard]:
    """Load all WorkerCards from .md files in *directory*.

    Only files with a ``.md`` suffix are processed; other files are silently
    skipped.  Files are sorted by name for deterministic ordering.

    Raises:
        ValueError: Propagated from :func:`parse_worker_md` for any invalid
                    worker definition file.
    """
    workers: list[WorkerCard] = []
    for md_file in sorted(directory.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        workers.append(parse_worker_md(content))
    return workers
