from __future__ import annotations
import pytest
from neoagent.tools.deferred import DeferredToolRegistry, ToolIndex


# ── ToolIndex dataclass ──────────────────────────────────────────────────────

def test_toolindex_fields():
    idx = ToolIndex(name="github__create_issue", description="Create a GitHub issue")
    assert idx.name == "github__create_issue"
    assert idx.description == "Create a GitHub issue"


# ── register() ───────────────────────────────────────────────────────────────

def test_register_adds_to_deferred():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create a GitHub issue")
    assert reg.is_deferred("github__create_issue")


def test_register_multiple_tools():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create issue")
    reg.register("github__list_repos", "List repos")
    assert reg.is_deferred("github__create_issue")
    assert reg.is_deferred("github__list_repos")


def test_register_duplicate_no_error():
    """Re-registering the same tool must not raise (idempotent)."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.register("github__create_issue", "desc")  # no error
    assert reg.is_deferred("github__create_issue")


# ── is_deferred() ────────────────────────────────────────────────────────────

def test_is_deferred_unknown_tool_returns_false():
    reg = DeferredToolRegistry()
    assert not reg.is_deferred("nonexistent__tool")


def test_is_deferred_after_promote_returns_false():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.promote({"github__create_issue"})
    assert not reg.is_deferred("github__create_issue")


# ── promote() ────────────────────────────────────────────────────────────────

def test_promote_removes_from_deferred():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.register("github__list_repos", "desc")
    reg.promote({"github__create_issue"})
    assert not reg.is_deferred("github__create_issue")
    assert reg.is_deferred("github__list_repos")


def test_promote_subset():
    reg = DeferredToolRegistry()
    reg.register("a__tool1", "t1")
    reg.register("a__tool2", "t2")
    reg.register("a__tool3", "t3")
    reg.promote({"a__tool1", "a__tool3"})
    assert not reg.is_deferred("a__tool1")
    assert reg.is_deferred("a__tool2")
    assert not reg.is_deferred("a__tool3")


def test_promote_nonexistent_no_error():
    """promote() of unknown tool name must not raise."""
    reg = DeferredToolRegistry()
    reg.promote({"nonexistent__tool"})  # should not raise


def test_promote_empty_set_no_error():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.promote(set())
    assert reg.is_deferred("github__create_issue")


# ── get_deferred_names() ──────────────────────────────────────────────────────

def test_get_deferred_names_empty():
    reg = DeferredToolRegistry()
    assert reg.get_deferred_names() == []


def test_get_deferred_names_returns_all_deferred():
    reg = DeferredToolRegistry()
    reg.register("a__t1", "d1")
    reg.register("a__t2", "d2")
    names = reg.get_deferred_names()
    assert set(names) == {"a__t1", "a__t2"}


def test_get_deferred_names_excludes_promoted():
    reg = DeferredToolRegistry()
    reg.register("a__t1", "d1")
    reg.register("a__t2", "d2")
    reg.promote({"a__t1"})
    names = reg.get_deferred_names()
    assert names == ["a__t2"]


# ── reset() ───────────────────────────────────────────────────────────────────

def test_reset_clears_all():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.promote({"github__create_issue"})  # now promoted
    reg.register("github__list_repos", "desc")
    reg.reset()
    assert reg.get_deferred_names() == []
    assert not reg.is_deferred("github__list_repos")


def test_reset_then_register_works():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.reset()
    reg.register("github__create_issue", "desc again")
    assert reg.is_deferred("github__create_issue")


# ── search(): "select:name1,name2" mode ──────────────────────────────────────

def test_search_select_exact_match():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create a GitHub issue")
    reg.register("github__list_repos", "List repos")
    results = reg.search("select:github__create_issue")
    assert len(results) == 1
    assert results[0].name == "github__create_issue"


def test_search_select_multiple():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.register("github__list_repos", "desc")
    reg.register("github__delete_repo", "desc")
    results = reg.search("select:github__create_issue,github__list_repos")
    names = {r.name for r in results}
    assert names == {"github__create_issue", "github__list_repos"}


def test_search_select_missing_name_excluded():
    """select: for a name that doesn't exist returns no entry for that name."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    results = reg.search("select:github__create_issue,nonexistent__tool")
    assert len(results) == 1
    assert results[0].name == "github__create_issue"


def test_search_select_already_promoted_excluded():
    """select: should NOT return promoted tools (they are no longer deferred)."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.promote({"github__create_issue"})
    results = reg.search("select:github__create_issue")
    assert results == []


# ── search(): "+keyword rest" mode ───────────────────────────────────────────

def test_search_plus_keyword_filters_by_name():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create an issue on GitHub")
    reg.register("github__list_repos", "List GitHub repositories")
    reg.register("filesystem__read_file", "Read a file from disk")
    results = reg.search("+github create")
    names = {r.name for r in results}
    # Only tools whose name contains "github"
    assert "filesystem__read_file" not in names
    assert "github__create_issue" in names


def test_search_plus_keyword_name_filter_case_insensitive():
    reg = DeferredToolRegistry()
    reg.register("GitHub__Create_Issue", "desc")
    results = reg.search("+github anything")
    assert len(results) == 1


# ── search(): "keyword" (regex) mode ─────────────────────────────────────────

def test_search_keyword_matches_name():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create a GitHub issue")
    reg.register("filesystem__read_file", "Read a file")
    results = reg.search("issue")
    names = {r.name for r in results}
    assert "github__create_issue" in names
    assert "filesystem__read_file" not in names


def test_search_keyword_matches_description():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create a GitHub issue")
    reg.register("filesystem__read_file", "Read a file from disk")
    results = reg.search("disk")
    names = {r.name for r in results}
    assert "filesystem__read_file" in names
    assert "github__create_issue" not in names


def test_search_keyword_case_insensitive():
    reg = DeferredToolRegistry()
    reg.register("github__CREATE_ISSUE", "desc")
    results = reg.search("create")
    assert len(results) == 1


def test_search_keyword_no_match_returns_empty():
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create an issue")
    results = reg.search("xyz_no_match")
    assert results == []


def test_search_empty_registry_returns_empty():
    reg = DeferredToolRegistry()
    assert reg.search("anything") == []
    assert reg.search("select:anything") == []


def test_search_only_returns_deferred_tools():
    """search() must never return promoted tools."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create issue")
    reg.promote({"github__create_issue"})
    results = reg.search("issue")
    assert results == []


# ── remove() ──────────────────────────────────────────────────────────────────

def test_remove_erases_deferred_entry():
    """remove() must delete tools from the index so they are no longer searchable."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create an issue")
    reg.register("github__list_repos", "List repos")
    reg.remove({"github__create_issue"})
    assert not reg.is_deferred("github__create_issue")
    # Other tools must be unaffected
    assert reg.is_deferred("github__list_repos")


def test_remove_also_erases_from_all_index():
    """remove() must clear the _all index so search cannot find ghost entries."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "Create an issue")
    reg.remove({"github__create_issue"})
    # search must not find the removed tool even via select:
    results = reg.search("select:github__create_issue")
    assert results == []
    results = reg.search("issue")
    assert results == []


def test_remove_promoted_tool_cleans_up():
    """remove() must work for tools that were already promoted (not in _deferred)."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.promote({"github__create_issue"})
    # Tool is in _all but not in _deferred; remove should not raise
    reg.remove({"github__create_issue"})
    results = reg.search("select:github__create_issue")
    assert results == []


def test_remove_nonexistent_no_error():
    """remove() of an unknown tool name must not raise."""
    reg = DeferredToolRegistry()
    reg.remove({"totally__unknown"})  # no error


def test_remove_empty_set_no_error():
    """remove() with an empty set must not change state."""
    reg = DeferredToolRegistry()
    reg.register("github__create_issue", "desc")
    reg.remove(set())
    assert reg.is_deferred("github__create_issue")


def test_remove_multiple_tools():
    """remove() must handle removing multiple tools at once."""
    reg = DeferredToolRegistry()
    reg.register("a__t1", "desc1")
    reg.register("a__t2", "desc2")
    reg.register("a__t3", "desc3")
    reg.remove({"a__t1", "a__t3"})
    assert not reg.is_deferred("a__t1")
    assert reg.is_deferred("a__t2")
    assert not reg.is_deferred("a__t3")
