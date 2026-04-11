from __future__ import annotations
import pytest
from pathlib import Path
from neoagent.tools.pathguard import validate_path


def test_valid_path_in_allowed_dir(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hi")
    result = validate_path(str(f), [tmp_path])
    assert result == f.resolve()


def test_path_traversal_blocked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside allowed"):
        validate_path("/etc/passwd", [tmp_path])


def test_dotdot_traversal_blocked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside allowed"):
        validate_path(str(tmp_path / ".." / ".." / "etc" / "passwd"), [tmp_path])


def test_multiple_allowed_dirs(tmp_path: Path) -> None:
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    f = d2 / "file.txt"
    f.write_text("ok")
    result = validate_path(str(f), [d1, d2])
    assert result == f.resolve()


def test_nonexistent_file_in_allowed_dir(tmp_path: Path) -> None:
    # validate_path should succeed even if the file does not yet exist,
    # as long as the resolved path falls inside an allowed directory.
    future_file = tmp_path / "new_file.txt"
    result = validate_path(str(future_file), [tmp_path])
    assert result == future_file.resolve()


def test_path_inside_subdirectory(tmp_path: Path) -> None:
    sub = tmp_path / "sub" / "nested"
    sub.mkdir(parents=True)
    f = sub / "data.txt"
    f.write_text("data")
    result = validate_path(str(f), [tmp_path])
    assert result == f.resolve()
