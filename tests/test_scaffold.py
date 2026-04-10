from __future__ import annotations


def test_package_importable() -> None:
    import neoagent  # noqa: F401
    assert True


def test_core_importable() -> None:
    import neoagent.core  # noqa: F401
    assert True


def test_tools_importable() -> None:
    import neoagent.tools  # noqa: F401
    assert True


def test_providers_importable() -> None:
    import neoagent.providers  # noqa: F401
    assert True
