from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from neoagent.tools.base import BaseTool
from neoagent.core.types import ToolResult

_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class WriteSkillInput(BaseModel):
    skill_name: str   # kebab-case, e.g. "xiumi-pattern"
    description: str  # one-line summary used in the skills index
    content: str      # markdown body without frontmatter; tool adds frontmatter


class WriteSkillTool(BaseTool):
    """Persist a newly discovered extraction pattern as a reusable learned skill.

    Call this after successfully exploring an unfamiliar page type with run_python.
    The written skill appears in the agent's skill list on the next run, allowing
    direct reuse without re-exploration.

    When to call:
      After run_python exploration succeeds for a new site pattern (e.g. SPA with
      embedded __NEXT_DATA__ JSON, image-only Xiumi articles, gzip-compressed API
      responses). Do NOT call for one-off pages — only when the pattern is likely
      to recur across many URLs on the same site/platform.

    skill_name (kebab-case, a-z 0-9 - only):
      Name by site or pattern, e.g. "xiumi-pattern", "mafengwo-itinerary", "ctrip-detail".

    description:
      One sentence describing when to use this skill, e.g.
      "Extract article text from Xiumi image-layout pages via embedded JSON API."

    content (markdown, no frontmatter):
      Must include:
        - Applicable conditions (URL patterns or HTML fingerprints like __NEXT_DATA__)
        - Step-by-step extraction procedure
        - Reference code snippet (copy-paste ready for run_python)
        - Success criteria (what a correct result looks like)

    Returns {status, path, name} on success, or {error} on failure.
    """

    name: str = "write_skill"
    description: str = (
        "Save a new extraction pattern as a learned skill so future agents can reuse it. "
        "Call after successfully exploring an unknown page type with run_python. "
        "skill_name must be kebab-case (a-z, 0-9, hyphens). "
        "content should include: applicable conditions, extraction steps, code snippet, success criteria. "
        "Returns {status, path, name}."
    )
    input_model: type[BaseModel] = WriteSkillInput
    permission: str = "auto"
    is_concurrent_safe: bool = True

    def __init__(self, learned_dir: Path) -> None:
        self._learned_dir = Path(learned_dir)

    async def execute(self, input: BaseModel) -> ToolResult:
        assert isinstance(input, WriteSkillInput)

        if not _KEBAB_RE.match(input.skill_name):
            return ToolResult(
                call_id="",
                output=json.dumps({
                    "error": (
                        f"Invalid skill_name '{input.skill_name}'. "
                        "Must be kebab-case: only lowercase letters, digits, and hyphens."
                    )
                }),
                is_error=True,
            )

        skill_dir = self._learned_dir / input.skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        frontmatter = f"---\nname: {input.skill_name}\ndescription: {input.description}\n---\n\n"
        skill_file.write_text(frontmatter + input.content, encoding="utf-8")

        return ToolResult(
            call_id="",
            output=json.dumps({
                "status": "written",
                "path": str(skill_file),
                "name": input.skill_name,
            }),
        )
