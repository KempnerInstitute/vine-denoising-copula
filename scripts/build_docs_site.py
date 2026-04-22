#!/usr/bin/env python3
"""Build a static GitHub Pages site from the docs folder."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
DEFAULT_OUT = REPO_ROOT / "site"

NAV_ITEMS = [
    ("Home", Path("index.html")),
    ("Readme", Path("readme.html")),
    ("Getting Started", Path("GETTING_STARTED.html")),
    ("User Guide", Path("USER_GUIDE.html")),
    ("API", Path("API.html")),
    ("Configuration", Path("CONFIGURATION.html")),
    ("Model Releases", Path("MODEL_RELEASES.html")),
]


def repo_web_base() -> str:
    try:
        raw = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "https://github.com/KempnerInstitute/vine-denoising-copula"

    if raw.startswith("git@github.com:"):
        path = raw.split(":", 1)[1]
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://github.com/{path}"
    if raw.startswith("https://github.com/"):
        return raw[:-4] if raw.endswith(".git") else raw
    return "https://github.com/KempnerInstitute/vine-denoising-copula"


def rel_link(current: Path, target: Path) -> str:
    return Path(
        Path(
            Path(*([".."] * len(current.parent.parts))) / target
            if current.parent.parts
            else target
        )
    ).as_posix()


def git_blob_url(repo_base: str, repo_rel: str) -> str:
    return f"{repo_base}/blob/main/{repo_rel.lstrip('./')}"


def resolve_internal_target(source: Path, raw_url: str) -> Path | None:
    raw = raw_url.split("#", 1)[0]
    if not raw:
        return None
    if raw == "README.md":
        return Path("readme.html")
    if raw.startswith("docs/"):
        p = Path(raw[5:])
        return p.with_suffix(".html") if p.suffix == ".md" else p
    if source == REPO_ROOT / "README.md":
        p = Path(raw)
        if p.parts and p.parts[0] == "docs":
            p = Path(*p.parts[1:])
            return p.with_suffix(".html") if p.suffix == ".md" else p
        return None
    base = source.parent
    candidate = (base / raw).resolve()
    try:
        rel = candidate.relative_to(DOCS_ROOT.resolve())
    except Exception:
        return None
    return rel.with_suffix(".html") if rel.suffix == ".md" else rel


def rewrite_url(raw_url: str, source: Path, current_out_rel: Path, repo_base: str) -> str:
    if raw_url.startswith(("http://", "https://", "mailto:", "#")):
        return raw_url

    anchor = ""
    if "#" in raw_url:
        raw_no_anchor, anchor = raw_url.split("#", 1)
        anchor = f"#{anchor}"
    else:
        raw_no_anchor = raw_url

    if raw_no_anchor == "../README.md":
        return rel_link(current_out_rel, Path("readme.html")) + anchor

    internal = resolve_internal_target(source, raw_no_anchor)
    if internal is not None:
        return rel_link(current_out_rel, internal) + anchor

    repo_rel = raw_no_anchor.lstrip("./")
    return git_blob_url(repo_base, repo_rel) + anchor


def rewrite_markdown_links(text: str, source: Path, current_out_rel: Path, repo_base: str) -> str:
    pattern = re.compile(r'(!?\[[^\]]*\])\(([^)]+)\)')

    def repl(match: re.Match[str]) -> str:
        label, url = match.groups()
        new_url = rewrite_url(url.strip(), source, current_out_rel, repo_base)
        return f"{label}({new_url})"

    return pattern.sub(repl, text)


def run_pandoc(markdown_text: str) -> str:
    proc = subprocess.run(
        ["pandoc", "--from=gfm", "--to=html5", "--no-highlight"],
        input=markdown_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def wrap_doc_page(*, title: str, body_html: str, current_out_rel: Path, repo_base: str) -> str:
    css_href = rel_link(current_out_rel, Path("assets/site.css"))
    index_href = rel_link(current_out_rel, Path("index.html"))
    repo_href = repo_base

    nav_html = "\n".join(
        f'<li><a href="{html.escape(rel_link(current_out_rel, target))}">{html.escape(label)}</a></li>'
        for label, target in NAV_ITEMS
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)} | VDC Docs</title>
    <link rel="stylesheet" href="{html.escape(css_href)}">
  </head>
  <body class="docs-page">
    <div class="doc-shell">
      <aside class="doc-sidebar">
        <p class="eyebrow">Vine Denoising Copula</p>
        <h1>{html.escape(title)}</h1>
        <p class="sidebar-copy">Usage, release, and verification documentation for the VDC codebase.</p>
        <nav>
          <ul>
            {nav_html}
            <li><a href="{html.escape(repo_href)}">GitHub Repo</a></li>
          </ul>
        </nav>
        <p class="sidebar-foot"><a href="{html.escape(index_href)}">Back to docs home</a></p>
      </aside>
      <main class="doc-content">
        <article class="markdown-body">
          {body_html}
        </article>
      </main>
    </div>
  </body>
</html>
"""


def build_markdown_page(source: Path, out_root: Path, repo_base: str) -> None:
    if source == REPO_ROOT / "README.md":
        out_rel = Path("readme.html")
    else:
        out_rel = source.relative_to(DOCS_ROOT).with_suffix(".html")
    out_path = out_root / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = source.read_text()
    if source == REPO_ROOT / "README.md":
        raw = raw.replace('src="docs/', 'src="')
        raw = raw.replace("src='docs/", "src='")
    rewritten = rewrite_markdown_links(raw, source, out_rel, repo_base)
    body_html = run_pandoc(rewritten)
    title = first_heading(raw, source.stem.replace("_", " "))
    wrapped = wrap_doc_page(title=title, body_html=body_html, current_out_rel=out_rel, repo_base=repo_base)
    out_path.write_text(wrapped)


def patch_index_html(source: Path, out_root: Path, repo_base: str) -> None:
    text = source.read_text()

    def repl(match: re.Match[str]) -> str:
        url = match.group(1)
        if url == "../README.md":
            new_url = "readme.html"
        elif url.endswith(".md"):
            target = Path(url)
            if url.startswith("reports/"):
                new_url = target.with_suffix(".html").as_posix()
            else:
                new_url = target.with_suffix(".html").name if len(target.parts) == 1 else target.with_suffix(".html").as_posix()
        else:
            new_url = url
        return f'href="{new_url}"'

    patched = re.sub(r'href="([^"]+)"', repl, text)
    out_path = out_root / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patched)


def copy_static_files(out_root: Path) -> None:
    for path in DOCS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(DOCS_ROOT)
        if rel == Path("index.html"):
            continue
        if path.suffix == ".md":
            continue
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (out_root / ".nojekyll").write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the VDC docs site for GitHub Pages")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_root = args.out_dir.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    repo_base = repo_web_base()

    copy_static_files(out_root)
    patch_index_html(DOCS_ROOT / "index.html", out_root, repo_base)
    build_markdown_page(REPO_ROOT / "README.md", out_root, repo_base)
    for md in sorted(DOCS_ROOT.rglob("*.md")):
        build_markdown_page(md, out_root, repo_base)

    print(out_root)


if __name__ == "__main__":
    main()
