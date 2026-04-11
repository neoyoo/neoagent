from __future__ import annotations
from pathlib import Path

_DEFAULT_ALLOWED = [Path.cwd()]


def validate_path(file_path: str, allowed_directories: list[Path]) -> Path:
    """Resolve a path and verify it is within an allowed directory.

    Raises ValueError if the path escapes all allowed directories.
    """
    resolved = Path(file_path).resolve()
    for allowed in allowed_directories:
        try:
            if resolved.is_relative_to(allowed.resolve()):
                return resolved
        except (ValueError, TypeError):
            continue
    raise ValueError(
        f"Path {file_path!r} is outside allowed directories: "
        f"{[str(d) for d in allowed_directories]}"
    )
