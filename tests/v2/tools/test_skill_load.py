# tests/v2/tools/test_skill_load.py
"""Unit tests for SkillLoadTool."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from neoagent.tools.builtin.skill_load import SkillLoadTool, SkillLoadInput


def _make_tool(skills_dir: Path, allowed: list[str] | None = None) -> SkillLoadTool:
    return SkillLoadTool(skills_dir=skills_dir, allowed=allowed)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


_FM = "---\nname: {name}\ndescription: {desc}\n---\n"


# ── 1. Flat layout ─────────────────────────────────────────────────────────────

class TestFlatLayout:
    @pytest.mark.asyncio
    async def test_flat_skill_returns_body_only(self, tmp_path):
        skill_file = tmp_path / "foo.md"
        _write(skill_file, _FM.format(name="foo", desc="A foo skill") + "body content here")

        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="foo"))

        assert result.is_error is False
        assert result.output == "body content here"

    @pytest.mark.asyncio
    async def test_flat_skill_frontmatter_stripped(self, tmp_path):
        _write(tmp_path / "bar.md", _FM.format(name="bar", desc="desc") + "## Steps\nDo stuff.")
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="bar"))
        assert "---" not in result.output
        assert "## Steps" in result.output


# ── 2. Directory layout ────────────────────────────────────────────────────────

class TestDirectoryLayout:
    @pytest.mark.asyncio
    async def test_directory_skill_loaded(self, tmp_path):
        _write(
            tmp_path / "bar" / "SKILL.md",
            _FM.format(name="bar", desc="A bar skill") + "bar body",
        )
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="bar"))
        assert result.is_error is False
        assert result.output == "bar body"


# ── 3. Learned layout ─────────────────────────────────────────────────────────

class TestLearnedLayout:
    @pytest.mark.asyncio
    async def test_learned_skill_loaded(self, tmp_path):
        _write(
            tmp_path / "learned" / "baz" / "SKILL.md",
            _FM.format(name="baz", desc="A learned skill") + "baz body",
        )
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="baz"))
        assert result.is_error is False
        assert result.output == "baz body"


# ── 4. Skill without frontmatter ──────────────────────────────────────────────

class TestNoFrontmatter:
    @pytest.mark.asyncio
    async def test_no_frontmatter_returns_full_content(self, tmp_path):
        raw = "# No Frontmatter\nJust the body."
        _write(tmp_path / "plain.md", raw)
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="plain"))
        assert result.is_error is False
        assert result.output == raw

    def test_no_frontmatter_list_skills_name_is_stem(self, tmp_path):
        _write(tmp_path / "plain.md", "# No Frontmatter")
        tool = _make_tool(tmp_path)
        skills = tool.list_skills()
        names = [s["name"] for s in skills]
        assert "plain" in names

    def test_no_frontmatter_description_empty(self, tmp_path):
        _write(tmp_path / "plain.md", "# No Frontmatter")
        tool = _make_tool(tmp_path)
        skills = tool.list_skills()
        plain = next(s for s in skills if s["name"] == "plain")
        assert plain["description"] == ""


# ── 5. Skill not found ────────────────────────────────────────────────────────

class TestSkillNotFound:
    @pytest.mark.asyncio
    async def test_not_found_returns_error_json(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="missing"))
        assert result.is_error is False  # tool returns error in JSON, not is_error
        data = json.loads(result.output)
        assert "error" in data
        assert "available_skills" in data

    @pytest.mark.asyncio
    async def test_not_found_error_message_mentions_skill_name(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="ghost"))
        data = json.loads(result.output)
        assert "ghost" in data["error"]


# ── 6. Whitelist: skill not in whitelist → error JSON ─────────────────────────

class TestWhitelistBlocked:
    @pytest.mark.asyncio
    async def test_blocked_by_whitelist_returns_error_json(self, tmp_path):
        _write(tmp_path / "secret.md", _FM.format(name="secret", desc="hidden") + "body")
        _write(tmp_path / "allowed.md", _FM.format(name="allowed", desc="ok") + "body")
        tool = _make_tool(tmp_path, allowed=["allowed"])
        result = await tool.execute(SkillLoadInput(skill_name="secret"))
        data = json.loads(result.output)
        assert "error" in data
        assert "available_skills" in data
        assert "secret" not in data["available_skills"]

    @pytest.mark.asyncio
    async def test_blocked_available_skills_only_shows_whitelisted(self, tmp_path):
        _write(tmp_path / "visible.md", _FM.format(name="visible", desc="v") + "body")
        _write(tmp_path / "hidden.md", _FM.format(name="hidden", desc="h") + "body")
        tool = _make_tool(tmp_path, allowed=["visible"])
        result = await tool.execute(SkillLoadInput(skill_name="hidden"))
        data = json.loads(result.output)
        assert "visible" in data["available_skills"]
        assert "hidden" not in data["available_skills"]


# ── 7. Whitelist: learned skill bypasses whitelist ────────────────────────────

class TestLearnedBypassesWhitelist:
    @pytest.mark.asyncio
    async def test_learned_skill_bypasses_whitelist(self, tmp_path):
        _write(
            tmp_path / "learned" / "my-pattern" / "SKILL.md",
            _FM.format(name="my-pattern", desc="learned") + "learned body",
        )
        tool = _make_tool(tmp_path, allowed=["something-else"])
        result = await tool.execute(SkillLoadInput(skill_name="my-pattern"))
        assert result.is_error is False
        assert result.output == "learned body"


# ── 8. list_skills() without whitelist returns all ────────────────────────────

class TestListSkillsNoWhitelist:
    def test_list_all_visible_skills(self, tmp_path):
        _write(tmp_path / "flat.md", _FM.format(name="flat", desc="f") + "body")
        _write(tmp_path / "dir-skill" / "SKILL.md", _FM.format(name="dir-skill", desc="d") + "body")
        _write(tmp_path / "learned" / "learnt" / "SKILL.md", _FM.format(name="learnt", desc="l") + "body")
        tool = _make_tool(tmp_path)
        skills = tool.list_skills()
        names = {s["name"] for s in skills}
        assert "flat" in names
        assert "dir-skill" in names
        assert "learnt" in names


# ── 9. list_skills() with whitelist → only whitelisted + learned ───────────────

class TestListSkillsWithWhitelist:
    def test_whitelist_filters_non_learned(self, tmp_path):
        _write(tmp_path / "allowed.md", _FM.format(name="allowed", desc="a") + "body")
        _write(tmp_path / "blocked.md", _FM.format(name="blocked", desc="b") + "body")
        _write(tmp_path / "learned" / "auto" / "SKILL.md", _FM.format(name="auto", desc="l") + "body")
        tool = _make_tool(tmp_path, allowed=["allowed"])
        skills = tool.list_skills()
        names = {s["name"] for s in skills}
        assert "allowed" in names
        assert "auto" in names
        assert "blocked" not in names


# ── 10. YAML parse error → graceful fallback ──────────────────────────────────

class TestMalformedFrontmatter:
    @pytest.mark.asyncio
    async def test_malformed_yaml_falls_back_gracefully(self, tmp_path):
        bad_fm = "---\n: invalid: yaml: [\n---\n"
        full = bad_fm + "actual body"
        _write(tmp_path / "broken.md", full)
        tool = _make_tool(tmp_path)
        result = await tool.execute(SkillLoadInput(skill_name="broken"))
        # Should not raise; returns the body (content after frontmatter or full raw)
        assert result.is_error is False

    def test_malformed_yaml_list_skills_name_is_stem(self, tmp_path):
        bad_fm = "---\n: invalid: yaml: [\n---\n"
        _write(tmp_path / "broken.md", bad_fm + "body")
        tool = _make_tool(tmp_path)
        skills = tool.list_skills()
        names = [s["name"] for s in skills]
        assert "broken" in names

    def test_malformed_yaml_description_empty(self, tmp_path):
        bad_fm = "---\n: invalid: yaml: [\n---\n"
        _write(tmp_path / "broken.md", bad_fm + "body")
        tool = _make_tool(tmp_path)
        skills = tool.list_skills()
        broken = next(s for s in skills if s["name"] == "broken")
        assert broken["description"] == ""
