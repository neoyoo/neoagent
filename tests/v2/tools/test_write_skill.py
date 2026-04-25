# tests/v2/tools/test_write_skill.py
"""Unit tests for WriteSkillTool."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from neoagent.tools.builtin.write_skill import WriteSkillTool, WriteSkillInput


def _make_tool(learned_dir: Path) -> WriteSkillTool:
    return WriteSkillTool(learned_dir=learned_dir)


# ── 1. Valid kebab-case name → file created at expected path ──────────────────

class TestValidName:
    @pytest.mark.asyncio
    async def test_valid_name_creates_file(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="my-skill",
            description="Does something useful",
            content="## Steps\nDo it.",
        ))
        assert result.is_error is False
        expected = tmp_path / "my-skill" / "SKILL.md"
        assert expected.exists()

    @pytest.mark.asyncio
    async def test_valid_name_with_numbers(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="skill42",
            description="Has numbers",
            content="body",
        ))
        assert result.is_error is False
        assert (tmp_path / "skill42" / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_single_word_valid(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="foo",
            description="simple",
            content="bar",
        ))
        assert result.is_error is False


# ── 2. Invalid: uppercase → is_error=True ─────────────────────────────────────

class TestInvalidUppercase:
    @pytest.mark.asyncio
    async def test_uppercase_returns_error(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="FooBar",
            description="bad",
            content="body",
        ))
        assert result.is_error is True
        data = json.loads(result.output)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_error_message_mentions_kebab_case(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="FooBar",
            description="bad",
            content="body",
        ))
        data = json.loads(result.output)
        assert "kebab" in data["error"].lower()


# ── 3. Invalid: underscore → is_error=True ────────────────────────────────────

class TestInvalidUnderscore:
    @pytest.mark.asyncio
    async def test_underscore_returns_error(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="foo_bar",
            description="bad",
            content="body",
        ))
        assert result.is_error is True


# ── 4. Invalid: leading hyphen → is_error=True ────────────────────────────────

class TestInvalidLeadingHyphen:
    @pytest.mark.asyncio
    async def test_leading_hyphen_returns_error(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="-foo",
            description="bad",
            content="body",
        ))
        assert result.is_error is True


# ── 5. Invalid: empty name → is_error=True ────────────────────────────────────

class TestInvalidEmptyName:
    @pytest.mark.asyncio
    async def test_empty_name_returns_error(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="",
            description="empty",
            content="body",
        ))
        assert result.is_error is True


# ── 6. Return JSON has status + path + name ────────────────────────────────────

class TestReturnPayload:
    @pytest.mark.asyncio
    async def test_return_json_fields(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(WriteSkillInput(
            skill_name="my-skill",
            description="desc",
            content="body",
        ))
        data = json.loads(result.output)
        assert data["status"] == "written"
        assert data["name"] == "my-skill"
        expected_path = str(tmp_path / "my-skill" / "SKILL.md")
        assert data["path"] == expected_path


# ── 7. Re-writing same skill name overwrites ──────────────────────────────────

class TestOverwrite:
    @pytest.mark.asyncio
    async def test_overwrite_replaces_content(self, tmp_path):
        tool = _make_tool(tmp_path)
        await tool.execute(WriteSkillInput(
            skill_name="overwrite-me",
            description="v1",
            content="first content",
        ))
        await tool.execute(WriteSkillInput(
            skill_name="overwrite-me",
            description="v2",
            content="second content",
        ))
        text = (tmp_path / "overwrite-me" / "SKILL.md").read_text(encoding="utf-8")
        assert "second content" in text
        assert "first content" not in text


# ── 8. Frontmatter format exact ───────────────────────────────────────────────

class TestFrontmatterFormat:
    @pytest.mark.asyncio
    async def test_exact_frontmatter_format(self, tmp_path):
        tool = _make_tool(tmp_path)
        await tool.execute(WriteSkillInput(
            skill_name="exact-fmt",
            description="My description",
            content="## Body\nContent here.",
        ))
        text = (tmp_path / "exact-fmt" / "SKILL.md").read_text(encoding="utf-8")
        expected_fm = (
            "---\n"
            "name: exact-fmt\n"
            "description: My description\n"
            "---\n\n"
        )
        assert text.startswith(expected_fm)
        assert text == expected_fm + "## Body\nContent here."

    @pytest.mark.asyncio
    async def test_frontmatter_separator_is_triple_dash(self, tmp_path):
        tool = _make_tool(tmp_path)
        await tool.execute(WriteSkillInput(
            skill_name="sep-check",
            description="desc",
            content="body",
        ))
        text = (tmp_path / "sep-check" / "SKILL.md").read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == "---"
        assert lines[3] == "---"
