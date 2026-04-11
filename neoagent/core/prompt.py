from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass
class PromptSection:
    name: str
    content: str | Callable[[], str]
    priority: int
    is_static: bool = True

class PromptBuilder:
    def __init__(self):
        self._sections: list[PromptSection] = []
        self._skills: dict[str, PromptSection] = {}

    def add_section(self, section: PromptSection) -> None:
        if any(s.name == section.name for s in self._sections):
            raise ValueError(f"Section '{section.name}' already exists")
        self._sections.append(section)

    def remove_section(self, name: str) -> None:
        self._sections = [s for s in self._sections if s.name != name]

    def register_skill(self, name: str, section: PromptSection) -> None:
        """Register a skill section without activating it."""
        if name in self._skills:
            raise ValueError(f"Skill '{name}' already registered")
        self._skills[name] = section

    def activate_skill(self, name: str) -> None:
        """Move a registered skill into the active prompt sections."""
        if name not in self._skills:
            raise ValueError(f"Skill '{name}' is not registered")
        if not any(s.name == name for s in self._sections):
            self._sections.append(self._skills[name])

    def deactivate_skill(self, name: str) -> None:
        """Remove an active skill from the prompt (stays registered)."""
        self._sections = [s for s in self._sections if s.name != name]

    def is_skill_active(self, name: str) -> bool:
        return any(s.name == name for s in self._sections)

    def build(self) -> str:
        if not self._sections:
            return ""
        sorted_sections = sorted(
            self._sections,
            key=lambda s: (not s.is_static, s.priority)
        )
        parts = []
        for s in sorted_sections:
            content = s.content if isinstance(s.content, str) else s.content()
            parts.append(f"# {s.name}\n{content}")
        return "\n\n".join(parts)
