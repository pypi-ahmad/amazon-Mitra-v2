"""Verify required documentation, repository-relative Markdown links, and public API docs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "cache" / "docs_ok.json"
DOCUMENTS = {
    ROOT / "README.md": ("# Glass Box", "## Documentation"),
    ROOT / "CONTRIBUTING.md": ("# Contributing to Glass Box", "## Change workflow"),
    ROOT / "docs" / "DEVELOPER_GUIDE.md": ("# Developer guide", "## Local workflow"),
    ROOT / "docs" / "ONBOARDING.md": ("# Onboarding", "## First contribution"),
    ROOT / "docs" / "ZERO_TO_MASTERY.md": ("# Glass Box: zero to mastery", "## 1. Start Glass Box"),
    ROOT / "docs" / "PYTHON_REFERENCE.md": ("# Python reference", "## Model execution and results"),
}
LINK_PATTERN = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def main() -> None:
    """Run the static documentation contract without data or model downloads."""
    checked_links = _check_documents_and_links()
    documented = _check_public_src_docstrings()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {"documents": len(DOCUMENTS), "links": checked_links, "public_api": documented},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"documentation smoke passed: {documented} public APIs, {checked_links} local links")


def _check_documents_and_links() -> int:
    markdown_files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "STATUS.md"]
    markdown_files.extend(sorted((ROOT / "docs").glob("*.md")))
    for path, headings in DOCUMENTS.items():
        text = path.read_text(encoding="utf-8")
        missing = [heading for heading in headings if heading not in text]
        assert not missing, f"{path.relative_to(ROOT)} missing headings: {missing}"

    checked = 0
    for path in markdown_files:
        for raw_link in LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
            link = raw_link.strip().strip("<>")
            if not link or link.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local_path = link.split("#", 1)[0]
            if not local_path:
                continue
            assert (path.parent / local_path).resolve().exists(), (
                f"Broken local link in {path.relative_to(ROOT)}: {raw_link}"
            )
            checked += 1
    return checked


def _check_public_src_docstrings() -> int:
    documented = 0
    for path in sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert ast.get_docstring(tree), f"Missing module docstring: {path.relative_to(ROOT)}"
        for node in tree.body:
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                assert ast.get_docstring(node), (
                    f"Missing public docstring: {path.relative_to(ROOT)}::{node.name}"
                )
                documented += 1
    return documented


if __name__ == "__main__":
    main()
