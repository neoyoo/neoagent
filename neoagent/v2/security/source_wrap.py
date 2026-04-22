# neoagent/v2/security/source_wrap.py
"""Source content wrapping for external tool results.

Tools with returns_external_content=True have their output wrapped in
<source type="..." url="..."> ... </source> tags, which the LLM treats
as data (not instructions) per TAG_CONTRACT.

spec § 18.2 (lines 2938+), decision D from 2026-04-22 design freeze.
"""

from typing import Any


def wrap_source_content(
    output: str,
    tool_name: str,
    url: str | None = None,
    extra_attrs: dict[str, str] | None = None,
) -> str:
    """Wrap tool output in <source> tag.

    Args:
        output: The raw tool output string.
        tool_name: Used as type="..." attribute.
        url: Optional source URL (e.g. for web_fetch).
        extra_attrs: Additional attributes (e.g. size="18.3KB").

    Returns:
        Wrapped string like: <source type="web_fetch" url="...">...</source>
    """
    attrs = [f'type="{_escape_attr(tool_name)}"']
    if url is not None:
        attrs.append(f'url="{_escape_attr(url)}"')
    if extra_attrs:
        for k, v in extra_attrs.items():
            attrs.append(f'{_escape_attr(k)}="{_escape_attr(v)}"')
    attr_str = " ".join(attrs)
    return f"<source {attr_str}>\n{output}\n</source>"


def _escape_attr(s: str) -> str:
    """Minimal XML attribute escape (quotes/ampersand/angle brackets)."""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


async def source_wrap_hook(event: Any) -> Any:
    """post_tool_call hook that wraps output when tool.returns_external_content=True.

    Expected event attributes (based on HookManager's PostToolCallEvent):
    - tool: the BaseTool instance (has .returns_external_content attr)
    - result: ToolResult (mutable .output field)

    NOTE: This function is NOT registered in this batch. Phase 7 (NeoAgent
    init) registers via agent.hook("post_tool_call", source_wrap_hook).
    """
    tool = getattr(event, "tool", None)
    result = getattr(event, "result", None)
    if tool is None or result is None:
        return None  # malformed event; no-op
    if not getattr(tool, "returns_external_content", False):
        return None  # not external; no wrap
    if getattr(result, "is_error", False):
        return None  # error results not wrapped (preserve error context)

    # Wrap the output in place
    original = result.output
    # Extract optional url from tool input if available (generic via extra_attrs)
    # Kept minimal: only tool_name for now; tool-specific url handling can be
    # added by tools themselves calling wrap_source_content() directly.
    result.output = wrap_source_content(original, tool_name=tool.name)
    return None
