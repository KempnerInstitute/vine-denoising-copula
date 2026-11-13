"""Check user-facing documentation paths and site link translation."""
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_docs_site", ROOT / "scripts/build_docs_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source,url,expected", [
    ("docs/API.md", "../examples/", "tree/main/examples"),
    ("docs/RELEASING.md", "../CHANGELOG.md", "blob/main/CHANGELOG.md"),
    ("docs/API.md", "USER_GUIDE.md", "USER_GUIDE.html"),
    ("docs/API.md", "../README.md", "readme.html"),
    ("README.md", "docs/GETTING_STARTED.md", "GETTING_STARTED.html"),
])
def test_site_links_resolve_relative_to_source(source, url, expected):
    builder = load_builder()
    actual = builder.rewrite_url(url, ROOT / source, Path("API.html"), "https://example.org/repo")
    assert actual.endswith(expected)


def test_user_documentation_has_no_missing_local_targets():
    paths = [ROOT / name for name in ["README.md", "CHANGELOG.md", "CONTRIBUTING.md",
                                     "examples/README.md", "configs/README.md",
                                     "paper/README.md", "paper/figures/README.md",
                                     "paper/tables/README.md"]]
    paths += list((ROOT / "docs").glob("*.md"))
    missing = []
    for path in paths:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).exists():
                missing.append(f"{path.relative_to(ROOT)}: {target}")
    assert not missing, "\n".join(missing)
