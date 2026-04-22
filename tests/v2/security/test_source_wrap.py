"""Tests for Task 6.1 — source_wrap module.

spec § 18.2 (source tag wrapping semantics)
"""
from __future__ import annotations

import pytest

from neoagent.v2.security.source_wrap import source_wrap_hook, wrap_source_content


# ---------------------------------------------------------------------------
# wrap_source_content tests
# ---------------------------------------------------------------------------


def test_wrap_basic():
    """wrap_source_content produces correct tag with type attr only."""
    result = wrap_source_content("hello", "web_fetch")
    assert result == '<source type="web_fetch">\nhello\n</source>'


def test_wrap_with_url():
    """url attribute is included when provided."""
    result = wrap_source_content("hello", "web_fetch", url="https://x.com")
    assert 'url="https://x.com"' in result
    assert 'type="web_fetch"' in result
    assert "hello" in result


def test_wrap_with_extra_attrs():
    """extra_attrs are rendered as additional XML attributes."""
    result = wrap_source_content("content", "file_read", extra_attrs={"size": "1KB"})
    assert 'size="1KB"' in result
    assert "content" in result


def test_escape_in_url_attr():
    """Double-quotes in url are escaped to &quot;."""
    result = wrap_source_content("x", "web_fetch", url='say "hi"')
    assert "&quot;" in result
    assert '"hi"' not in result


def test_escape_in_tool_name():
    """Angle brackets in tool_name are escaped."""
    result = wrap_source_content("x", "tool<evil>")
    assert "&lt;" in result
    assert "&gt;" in result
    assert "<evil>" not in result


# ---------------------------------------------------------------------------
# source_wrap_hook tests
# ---------------------------------------------------------------------------


class _MockTool:
    def __init__(self, name: str, returns_external_content: bool = False):
        self.name = name
        self.returns_external_content = returns_external_content


class _MockResult:
    def __init__(self, output: str, is_error: bool = False):
        self.output = output
        self.is_error = is_error


class _MockEvent:
    def __init__(self, tool=None, result=None):
        if tool is not None:
            self.tool = tool
        if result is not None:
            self.result = result


@pytest.mark.asyncio
async def test_hook_no_tool():
    """Event without 'tool' attribute is a no-op; returns None."""
    event = _MockEvent(result=_MockResult("data"))
    ret = await source_wrap_hook(event)
    assert ret is None


@pytest.mark.asyncio
async def test_hook_tool_not_external():
    """Tool with returns_external_content=False → output unchanged."""
    tool = _MockTool("bash_run", returns_external_content=False)
    result = _MockResult("output data")
    event = _MockEvent(tool=tool, result=result)
    await source_wrap_hook(event)
    assert result.output == "output data"


@pytest.mark.asyncio
async def test_hook_tool_external_content():
    """Tool with returns_external_content=True → output is wrapped."""
    tool = _MockTool("web_fetch", returns_external_content=True)
    result = _MockResult("page content")
    event = _MockEvent(tool=tool, result=result)
    await source_wrap_hook(event)
    assert '<source type="web_fetch">' in result.output
    assert "page content" in result.output
    assert "</source>" in result.output


@pytest.mark.asyncio
async def test_hook_error_result_not_wrapped():
    """Error results are not wrapped even when tool.returns_external_content=True."""
    tool = _MockTool("web_fetch", returns_external_content=True)
    result = _MockResult("fetch error: timeout", is_error=True)
    event = _MockEvent(tool=tool, result=result)
    await source_wrap_hook(event)
    assert result.output == "fetch error: timeout"


@pytest.mark.asyncio
async def test_hook_wraps_in_place():
    """result.output is modified to wrapped content (in-place mutation)."""
    tool = _MockTool("file_read", returns_external_content=True)
    result = _MockResult("file contents here")
    event = _MockEvent(tool=tool, result=result)
    await source_wrap_hook(event)
    expected = wrap_source_content("file contents here", tool_name="file_read")
    assert result.output == expected
