from __future__ import annotations
import pytest
from neoagent.core.prompt import PromptSection, PromptBuilder

class TestPromptSection:
    def test_static_section(self):
        s = PromptSection(name="identity", content="You are neoagent.", priority=0, is_static=True)
        assert s.name == "identity"
        assert s.content == "You are neoagent."
        assert s.priority == 0
        assert s.is_static is True

    def test_dynamic_section_with_callable(self):
        s = PromptSection(name="env", content=lambda: "Python 3.11", priority=10, is_static=False)
        assert callable(s.content)

    def test_default_is_static(self):
        s = PromptSection(name="test", content="x", priority=0)
        assert s.is_static is True

class TestPromptBuilder:
    def test_empty_build(self):
        builder = PromptBuilder()
        assert builder.build() == ""

    def test_single_section(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="identity", content="You are neoagent.", priority=0))
        result = builder.build()
        assert "# identity" in result
        assert "You are neoagent." in result

    def test_static_before_dynamic(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="dynamic", content=lambda: "dynamic content", priority=0, is_static=False))
        builder.add_section(PromptSection(name="static", content="static content", priority=0, is_static=True))
        result = builder.build()
        static_pos = result.index("# static")
        dynamic_pos = result.index("# dynamic")
        assert static_pos < dynamic_pos

    def test_priority_ordering_within_same_type(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="low", content="low priority", priority=10, is_static=True))
        builder.add_section(PromptSection(name="high", content="high priority", priority=1, is_static=True))
        result = builder.build()
        high_pos = result.index("# high")
        low_pos = result.index("# low")
        assert high_pos < low_pos

    def test_callable_content_executed_on_build(self):
        call_count = 0
        def dynamic():
            nonlocal call_count
            call_count += 1
            return f"call {call_count}"
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="dyn", content=dynamic, priority=0, is_static=False))
        result1 = builder.build()
        result2 = builder.build()
        assert "call 1" in result1
        assert "call 2" in result2
        assert call_count == 2

    def test_remove_section(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="a", content="aaa", priority=0))
        builder.add_section(PromptSection(name="b", content="bbb", priority=1))
        builder.remove_section("a")
        result = builder.build()
        assert "aaa" not in result
        assert "bbb" in result

    def test_remove_nonexistent_no_error(self):
        builder = PromptBuilder()
        builder.remove_section("nope")  # should not raise

    def test_duplicate_name_raises(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="a", content="x", priority=0))
        with pytest.raises(ValueError):
            builder.add_section(PromptSection(name="a", content="y", priority=1))

    def test_sections_separated_by_blank_lines(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="a", content="aaa", priority=0))
        builder.add_section(PromptSection(name="b", content="bbb", priority=1))
        result = builder.build()
        assert "\n\n" in result

    def test_full_ordering_static_priority_then_dynamic_priority(self):
        builder = PromptBuilder()
        builder.add_section(PromptSection(name="d2", content=lambda: "d2", priority=5, is_static=False))
        builder.add_section(PromptSection(name="s2", content="s2", priority=5, is_static=True))
        builder.add_section(PromptSection(name="d1", content=lambda: "d1", priority=1, is_static=False))
        builder.add_section(PromptSection(name="s1", content="s1", priority=1, is_static=True))
        result = builder.build()
        # Expected order: s1(static,1) → s2(static,5) → d1(dynamic,1) → d2(dynamic,5)
        positions = {name: result.index(f"# {name}") for name in ["s1", "s2", "d1", "d2"]}
        assert positions["s1"] < positions["s2"] < positions["d1"] < positions["d2"]


# --- v2 skill lazy loading tests ---

def test_register_skill_does_not_appear_in_build() -> None:
    pb = PromptBuilder()
    section = PromptSection(name="skill_a", content="skill content", priority=10)
    pb.register_skill("skill_a", section)
    result = pb.build()
    assert "skill content" not in result


def test_activate_skill_appears_in_build() -> None:
    pb = PromptBuilder()
    section = PromptSection(name="skill_a", content="skill content", priority=10)
    pb.register_skill("skill_a", section)
    pb.activate_skill("skill_a")
    assert "skill content" in pb.build()


def test_deactivate_skill_removed_from_build() -> None:
    pb = PromptBuilder()
    section = PromptSection(name="skill_a", content="skill content", priority=10)
    pb.register_skill("skill_a", section)
    pb.activate_skill("skill_a")
    pb.deactivate_skill("skill_a")
    assert "skill content" not in pb.build()


def test_is_skill_active_returns_correct_state() -> None:
    pb = PromptBuilder()
    section = PromptSection(name="skill_a", content="x", priority=10)
    pb.register_skill("skill_a", section)
    assert not pb.is_skill_active("skill_a")
    pb.activate_skill("skill_a")
    assert pb.is_skill_active("skill_a")
    pb.deactivate_skill("skill_a")
    assert not pb.is_skill_active("skill_a")


def test_activate_unknown_skill_raises() -> None:
    pb = PromptBuilder()
    with pytest.raises(ValueError, match="not registered"):
        pb.activate_skill("unknown_skill")


def test_register_duplicate_skill_raises() -> None:
    pb = PromptBuilder()
    section = PromptSection(name="skill_a", content="x", priority=10)
    pb.register_skill("skill_a", section)
    with pytest.raises(ValueError, match="already registered"):
        pb.register_skill("skill_a", section)
