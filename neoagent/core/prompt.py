from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from neoagent.v2.schema import Batch, BatchMember, Layer, MemoryEntry, WorkingMemory


@dataclass
class PromptSection:
    name: str
    content: str | Callable[[], str]
    priority: int
    is_static: bool = True


# ---------------------------------------------------------------------------
# Ephemeral layer names — not rendered by build_system_prompt
# ---------------------------------------------------------------------------
_EPHEMERAL_LAYER_NAMES = {"working_memory", "compressed_history", "memory_context"}

# Static layer order (subset of Layer enum members that appear in system prompt)
_STATIC_LAYER_NAMES = ["identity", "persistent_memory", "capabilities", "security"]


class LayeredPromptBuilder:
    """spec § 15.1 + decision G (flat 7-layer).

    Separates static system-prompt layers (IDENTITY, PERSISTENT_MEMORY,
    CAPABILITIES, SECURITY) from ephemeral layers rendered per-turn via
    build_ephemeral().
    """

    def __init__(self) -> None:
        # Lazily import to avoid circular imports at module load time
        from neoagent.v2.schema import Layer

        # _sections: Layer → list[PromptSection] (unsorted; sorted at build time)
        self._sections: dict[str, list[PromptSection]] = {layer.value: [] for layer in Layer}
        # skill name → PromptSection (all registered skills, active or not)
        self._registered_skills: dict[str, PromptSection] = {}
        # names of currently active skills
        self._active_skills: set[str] = set()

    # ------------------------------------------------------------------
    # Layer section registration
    # ------------------------------------------------------------------

    def register_layer_section(self, layer: "Layer", section: PromptSection) -> None:
        """Register *section* into *layer*.  Same layer keeps sections sorted
        by priority descending at build time — insertion order does not matter.
        """
        self._sections[layer.value].append(section)

    def remove_layer_section(self, layer: "Layer", name: str) -> None:
        """Remove the section with *name* from *layer*.  Raises KeyError if not found."""
        bucket = self._sections[layer.value]
        for i, sec in enumerate(bucket):
            if sec.name == name:
                del bucket[i]
                return
        raise KeyError(f"Section '{name}' not found in layer '{layer.value}'")

    # ------------------------------------------------------------------
    # Skill helpers (backward-compat surface, routes to CAPABILITIES)
    # ------------------------------------------------------------------

    def register_skill(self, name: str, section: PromptSection) -> None:
        """Register *section* under skill *name* in Layer.CAPABILITIES.
        The skill is inactive by default and will not appear in build_system_prompt
        until activate_skill() is called.
        """
        from neoagent.v2.schema import Layer

        if name in self._registered_skills:
            raise ValueError(f"Skill '{name}' already registered")
        self._registered_skills[name] = section
        # Add to CAPABILITIES bucket but guard at render time via _active_skills
        self.register_layer_section(Layer.CAPABILITIES, section)

    def activate_skill(self, name: str) -> None:
        """Mark *name* as active so it is included in build_system_prompt."""
        if name not in self._registered_skills:
            raise KeyError(f"Skill '{name}' is not registered")
        self._active_skills.add(name)

    def deactivate_skill(self, name: str) -> None:
        """Remove *name* from active set (skill stays registered)."""
        self._active_skills.discard(name)

    def is_skill_active(self, name: str) -> bool:
        return name in self._active_skills

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _render_content(self, section: PromptSection) -> str:
        if callable(section.content):
            return section.content()
        return section.content

    def build_system_prompt(self) -> str:
        """Render static layers only (IDENTITY → PERSISTENT_MEMORY → CAPABILITIES → SECURITY).

        Within each layer sections are sorted by priority descending (highest first).
        Skill sections in CAPABILITIES are only included when active.
        """
        # Skill names: map section.name → skill name for active lookup
        _skill_section_names: dict[str, str] = {
            sec.name: skill_name
            for skill_name, sec in self._registered_skills.items()
        }

        parts: list[str] = []
        for layer_name in _STATIC_LAYER_NAMES:
            bucket = self._sections.get(layer_name, [])
            sorted_sections = sorted(bucket, key=lambda s: s.priority, reverse=True)
            for sec in sorted_sections:
                # For CAPABILITIES: skip skill sections that are not active
                if layer_name == "capabilities" and sec.name in _skill_section_names:
                    skill_name = _skill_section_names[sec.name]
                    if skill_name not in self._active_skills:
                        continue
                parts.append(self._render_content(sec))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # build_ephemeral
    # ------------------------------------------------------------------

    def build_ephemeral(
        self,
        wm: "WorkingMemory | None",
        batches: "list[Batch]",
        memory_entries: "list[MemoryEntry] | None" = None,
    ) -> str:
        """Render ephemeral context layers as XML.

        Blocks emitted (when non-empty):
          1. <working_memory>  — spec § 7.1
          2. <compressed_history> — spec § 7.2
          3. <memory-context> — spec § 7.4
        """
        blocks: list[str] = []

        if wm is not None:
            blocks.append(self._render_working_memory(wm))

        if batches:
            blocks.append(self._render_compressed_history(batches))

        if memory_entries:
            blocks.append(self._render_memory_context(memory_entries))

        return "\n".join(blocks)

    # ------------------------------------------------------------------
    # Private XML renderers
    # ------------------------------------------------------------------

    def _render_working_memory(self, wm: "WorkingMemory") -> str:
        lines: list[str] = [
            f'<working_memory version="{wm.version}" at_turn="{wm.at_turn}">',
            "  CONSTRAINTS_AND_PREFERENCES:",
        ]
        for item in wm.constraints_and_preferences:
            lines.append(f"    - {item}")

        lines += [
            "",
            "  PROGRESS:",
            f"    {wm.progress}",
            "",
            "  KEY_DECISIONS:",
        ]
        for item in wm.key_decisions:
            lines.append(f"    - {item}")

        lines += [
            "",
            "  RELEVANT_FILES:",
        ]
        for item in wm.relevant_files:
            lines.append(f"    - {item}")

        lines += [
            "",
            "  NEXT_STEPS:",
        ]
        for item in wm.next_steps:
            lines.append(f"    - {item}")

        lines += [
            "",
            "  CRITICAL_CONTEXT:",
            f"    {wm.critical_context}",
            "</working_memory>",
        ]
        return "\n".join(lines)

    def _render_compressed_history(self, batches: "list[Batch]") -> str:
        lines: list[str] = ["<compressed_history>"]
        for batch in batches:
            lines.append(
                f'  <batch id="{batch.batch_id}" turns="{batch.turns_from}-{batch.turns_to}">'
            )
            lines.append("    <summary>")
            for summary_line in batch.summary.splitlines():
                lines.append(f"      {summary_line}")
            lines.append("    </summary>")
            if batch.members:
                lines.append("    <recoverable>")
                for member in batch.members:
                    lines.append(
                        f'      <msg id="{member.id}" role="{member.role}" preview="{member.preview}" />'
                    )
                lines.append("    </recoverable>")
            lines.append("  </batch>")
        lines.append("</compressed_history>")
        return "\n".join(lines)

    def _render_memory_context(self, entries: "list[MemoryEntry]") -> str:
        lines: list[str] = ["<memory-context>"]
        for entry in entries:
            lines.append(
                f'  <entry type="{entry.type}" confidence="{entry.confidence}">'
                f"{entry.content}</entry>"
            )
        lines.append("</memory-context>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PromptBuilder — kept for backward compat; deprecated in v2.0
# ---------------------------------------------------------------------------


class PromptBuilder:
    """[DEPRECATED in v2.0] Use LayeredPromptBuilder instead.
    Kept for backward compat; will be removed in v3.0."""

    def __init__(self):
        warnings.warn(
            "PromptBuilder is deprecated; use LayeredPromptBuilder (Layer-aware) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
