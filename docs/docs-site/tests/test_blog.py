"""Contract tests for the blog hook: cards, anchors, timeline and warnings."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

DOCS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS_ROOT / "tools"))
import blog  # noqa: E402
import site_build  # noqa: E402
from manifest_fixtures import published_release  # noqa: E402


def _post(directory: Path, name: str, title: str, date: str) -> Path:
    path = directory / name
    path.write_text(
        "---\n"
        f"title: {title}\n"
        f"date: {date}\n"
        "authors: UCM Team\n"
        f"summary: Summary for {title}.\n"
        "cover: assets/images/blog/cover.png\n"
        "---\n\n"
        f"# {title}\n",
        encoding="utf-8",
    )
    return path


def _page(root: Path, locale: str, name: str = "index.md") -> SimpleNamespace:
    blog_dir = root / "docs" / locale / "blog"
    path = blog_dir / name
    slug = path.stem
    url = f"/{locale}/blog/" + ("" if slug == "index" else f"{slug}/")
    return SimpleNamespace(
        file=SimpleNamespace(
            src_uri=f"{locale}/blog/{name}",
            abs_src_path=str(path),
            name=name,
        ),
        url=url,
        meta={},
    )


def _fixture(root: Path, locale: str = "en") -> Path:
    blog_dir = root / "docs" / locale / "blog"
    blog_dir.mkdir(parents=True)
    (blog_dir / "index.md").write_text(
        "# Blog\n\n<!-- blog-list -->\n", encoding="utf-8"
    )
    _post(blog_dir, "2026-11-09-ucm.md", "First post", "2026-11-09")
    _post(blog_dir, "2026-10-01-second.md", "Second post", "2026-10-01")
    return blog_dir


def test_render_groups_posts_by_year_and_month(tmp_path):
    _fixture(tmp_path)
    page = _page(tmp_path, "en")
    output = blog._render(page)

    assert 'id="year-2026"' in output
    assert 'id="month-2026-11"' in output
    assert 'id="month-2026-10"' in output
    assert "First post" in output
    assert "Second post" in output
    assert output.count('class="ucm-blog-badge"') == 1


def test_timeline_exposes_years_months_and_active_post(tmp_path):
    _fixture(tmp_path)
    index = blog._timeline(_page(tmp_path, "en"))
    assert [entry["year"] for entry in index["years"]] == [2026]
    months = index["years"][0]["months"]
    assert [month["month"] for month in months] == [11, 10]
    assert all(month["active"] is False for month in months)

    article = blog._timeline(_page(tmp_path, "en", "2026-11-09-ucm.md"))
    active = article["years"][0]
    assert active["active"] is True
    assert active["months"][0]["active"] is True
    assert active["months"][1]["active"] is False


def test_missing_date_falls_back_with_warning(tmp_path, caplog):
    blog_dir = _fixture(tmp_path)
    (blog_dir / "2026-12-01-undated.md").write_text(
        "---\ntitle: Undated\n---\n\n# Undated\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING):
        records = blog._records(blog_dir, "/en/blog/")

    assert any("no valid 'date'" in record.message for record in caplog.records)
    assert any(record["title"] == "Undated" for record in records)


def test_missing_list_marker_warns_and_keeps_content(tmp_path, caplog):
    _fixture(tmp_path)
    page = _page(tmp_path, "en")

    with caplog.at_level(logging.WARNING):
        output = blog.on_page_content("<h1>Blog</h1>", page, None, None)

    assert output == "<h1>Blog</h1>"
    assert any(
        "missing the '<!-- blog-list -->' marker" in item.message
        for item in caplog.records
    )


def test_real_site_build_renders_blog_index_and_article(tmp_path):
    output = tmp_path / "site"
    manifest = published_release()[1]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    site_build.build_site(
        language="zh",
        site_dir=output,
        site_url="https://docs.example.invalid/zh/v0.9.3/",
        repository="example/ucm",
        ref="v0.9.3",
        manifest_path=manifest_path,
        docs_root=DOCS_ROOT,
        strict=True,
    )

    index = (output / "blog/index.html").read_text(encoding="utf-8")
    assert "ucm-blog-timeline" in index
    assert "ucm-blog-nav-trigger" in index
    assert "md-nav--primary" in index
    assert "2026-11-09-ucm" in index
    assert "UCM 官网博客正式上线" in index

    article = (output / "blog/2026-11-09-ucm/index.html").read_text(encoding="utf-8")
    assert "UCM 官网博客正式上线" in article
    assert "Unified Cache Manager（UCM）" in article
