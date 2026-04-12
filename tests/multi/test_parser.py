from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from neoagent.multi.parser import load_workers, parse_worker_md
from neoagent.multi.worker import WorkerCard


VALID_MD = textwrap.dedent("""\
    ---
    name: coder
    description: Writes Python code
    model: claude-sonnet-4-6
    tags:
      - python
      - code
    tools:
      - bash
      - read
    ---
    You are a skilled Python developer.
    Write clean, tested code.
""")


class TestParseWorkerMd:
    def test_valid_md_returns_worker_card(self) -> None:
        card = parse_worker_md(VALID_MD)
        assert isinstance(card, WorkerCard)

    def test_name_parsed(self) -> None:
        card = parse_worker_md(VALID_MD)
        assert card.name == "coder"

    def test_instruction_is_body(self) -> None:
        card = parse_worker_md(VALID_MD)
        assert "You are a skilled Python developer." in card.instruction
        assert "Write clean, tested code." in card.instruction

    def test_optional_model_present(self) -> None:
        card = parse_worker_md(VALID_MD)
        assert card.model == "claude-sonnet-4-6"

    def test_optional_model_absent(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: analyst
            description: Analyzes things
            ---
            You analyze stuff.
        """)
        card = parse_worker_md(md)
        assert card.model is None

    def test_optional_tags_absent(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: analyst
            description: Analyzes things
            ---
            You analyze stuff.
        """)
        card = parse_worker_md(md)
        assert card.tags == ()

    def test_optional_tools_absent(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: analyst
            description: Analyzes things
            ---
            You analyze stuff.
        """)
        card = parse_worker_md(md)
        assert card.tools == ()

    def test_missing_name_raises(self) -> None:
        md = textwrap.dedent("""\
            ---
            description: Analyzes things
            ---
            Some body.
        """)
        with pytest.raises(ValueError, match="name"):
            parse_worker_md(md)

    def test_no_frontmatter_raises(self) -> None:
        md = "Just some plain text without frontmatter."
        with pytest.raises(ValueError, match="frontmatter"):
            parse_worker_md(md)

    def test_empty_body(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: minimal
            description: Minimal worker
            ---
        """)
        card = parse_worker_md(md)
        assert card.instruction.strip() == ""

    def test_malformed_yaml_raises(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: [broken
            description: bad yaml
            ---
            Body text.
        """)
        with pytest.raises(ValueError, match="[Yy][Aa][Mm][Ll]|[Mm]alformed|[Pp]arse"):
            parse_worker_md(md)

    def test_description_absent_defaults_empty(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: nodesc
            ---
            Some instruction.
        """)
        card = parse_worker_md(md)
        assert card.description == ""


class TestLoadWorkers:
    def test_load_from_directory(self, tmp_path: Path) -> None:
        md1 = textwrap.dedent("""\
            ---
            name: worker1
            description: First worker
            ---
            Instruction for worker 1.
        """)
        md2 = textwrap.dedent("""\
            ---
            name: worker2
            description: Second worker
            ---
            Instruction for worker 2.
        """)
        (tmp_path / "worker1.md").write_text(md1)
        (tmp_path / "worker2.md").write_text(md2)

        workers = load_workers(tmp_path)
        assert len(workers) == 2
        names = {w.name for w in workers}
        assert names == {"worker1", "worker2"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        workers = load_workers(tmp_path)
        assert workers == []

    def test_skip_non_md_files(self, tmp_path: Path) -> None:
        md = textwrap.dedent("""\
            ---
            name: realworker
            description: A worker
            ---
            Instructions.
        """)
        (tmp_path / "realworker.md").write_text(md)
        (tmp_path / "README.txt").write_text("Not a worker")
        (tmp_path / "config.yaml").write_text("key: value")
        (tmp_path / "script.py").write_text("print('hello')")

        workers = load_workers(tmp_path)
        assert len(workers) == 1
        assert workers[0].name == "realworker"

    def test_invalid_md_raises(self, tmp_path: Path) -> None:
        bad_md = "No frontmatter at all."
        (tmp_path / "bad.md").write_text(bad_md)
        with pytest.raises(ValueError):
            load_workers(tmp_path)
