from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_skill_file(path: Path) -> tuple[str, str, str]:
    """Return (name, description, content) from a SKILL.md with YAML frontmatter."""
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return path.stem, "", raw

    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}

    content = raw[m.end():]
    name = str(fm.get("name", path.stem))
    description = str(fm.get("description", "")).strip()
    return name, description, content


def _find_skill_file(skills_dir: Path, skill_name: str) -> Path | None:
    """Support flat, directory, and learned-subdirectory skill layouts."""
    candidates = [
        skills_dir / f"{skill_name}.md",
        skills_dir / skill_name / "SKILL.md",
        skills_dir / "learned" / skill_name / "SKILL.md",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _is_learned_skill(skills_dir: Path, skill_name: str) -> bool:
    """Return True if skill_name lives under the learned/ subdirectory."""
    learned_path = skills_dir / "learned" / skill_name / "SKILL.md"
    return learned_path.exists()


class SkillLoadInput(BaseModel):
    skill_name: str


class SkillLoadTool(BaseTool):
    """Built-in tool: load a skill's full content on demand (progressive disclosure).

    Skills are .md files with YAML frontmatter (name + description).
    Supports flat layout (skill.md) and directory layout (skill/SKILL.md).
    The agent sees only names + descriptions in the system prompt;
    full content enters the conversation only when explicitly requested.
    """

    name: str = "load_skill"
    description: str = (
        "Load a skill's full instructions by name. "
        "Returns the skill content for you to follow in this conversation. "
        "Call this before starting any task that has a matching skill."
    )
    input_model: type[BaseModel] = SkillLoadInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(
        self,
        skills_dir: Path,
        allowed: list[str] | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._allowed = set(allowed) if allowed is not None else None

    def list_skills(self) -> list[dict[str, str]]:
        """Return [{name, description}] for skills visible to this agent.

        Learned skills (under learned/) are always included regardless of the
        allowed whitelist — they were written by the agent itself and are always safe.
        """
        skills: list[dict[str, str]] = []
        seen: set[str] = set()

        for path in sorted(self._skills_dir.rglob("*.md")):
            if path.name == "SKILL.md" or path.parent == self._skills_dir:
                name, description, _ = _parse_skill_file(path)
                if name in seen:
                    continue
                # Skills under learned/ bypass the whitelist
                is_learned = (
                    path.parts[-2] != self._skills_dir.name
                    and "learned" in path.parts
                )
                if not is_learned and self._allowed is not None and name not in self._allowed:
                    continue
                skills.append({"name": name, "description": description})
                seen.add(name)
        return skills

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, SkillLoadInput)
        # Learned skills bypass the whitelist; check before applying allowed filter
        if (
            self._allowed is not None
            and input.skill_name not in self._allowed
            and not _is_learned_skill(self._skills_dir, input.skill_name)
        ):
            return ToolResult(
                call_id="",
                output=json.dumps({
                    "error": f"Skill '{input.skill_name}' not available to this agent",
                    "available_skills": [s["name"] for s in self.list_skills()],
                }),
            )
        skill_file = _find_skill_file(self._skills_dir, input.skill_name)
        if skill_file is None:
            available = [s["name"] for s in self.list_skills()]
            return ToolResult(
                call_id="",
                output=json.dumps({
                    "error": f"Skill '{input.skill_name}' not found",
                    "available_skills": available,
                }),
            )
        _, _, content = _parse_skill_file(skill_file)
        return ToolResult(call_id="", output=content)
