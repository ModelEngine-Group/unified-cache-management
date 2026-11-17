"""Build the blog landing page and timeline from post front matter.

A post is any Markdown file next to ``blog/index.md``. All posts are rendered
on one page grouped by year and month; the newest post gets the Latest badge.
The sidebar timeline is derived from the same records.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime
from pathlib import Path

import yaml
from mkdocs.utils import get_relative_url

LIST_MARKER = "<!-- blog-list -->"
FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
HEADING = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
LOG = logging.getLogger("mkdocs")

LABELS = {
    "en": {
        "latest": "Latest",
        "empty": "No posts yet.",
        "back": "All posts",
    },
    "zh": {
        "latest": "最新",
        "empty": "暂无文章，敬请期待。",
        "back": "返回博客",
    },
}


def _locale(src_uri: str) -> str:
    first = src_uri.split("/", 1)[0]
    return first if first in LABELS else "en"


def _parse(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[match.end() :]


def _date(value, path: Path) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass
    LOG.warning(
        f"Blog post '{path.name}' has no valid 'date' front matter; "
        "falling back to the file modification time."
    )
    return date.fromtimestamp(path.stat().st_mtime)


def _authors(meta: dict) -> str:
    value = meta.get("authors") or meta.get("author")
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value).strip() if value else ""


def _format_date(value: date, locale: str) -> str:
    if locale == "zh":
        return f"{value.year}年{value.month}月{value.day}日"
    return f"{value.strftime('%b')} {value.day}, {value.year}"


def _cover(value, page_url: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if "://" in value or value.startswith("/"):
        return value
    return get_relative_url(value, page_url)


def _title(meta: dict, body: str, path: Path) -> str:
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    found = HEADING.search(body)
    return found.group(1) if found else path.stem


def _records(directory: Path, page_url: str) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.md")):
        if path.name.startswith(("index", "_")):
            continue
        meta, body = _parse(path)
        if meta.get("draft") is True:
            continue
        records.append(
            {
                "slug": path.stem,
                "title": _title(meta, body, path),
                "date": _date(meta.get("date"), path),
                "authors": _authors(meta),
                "summary": str(
                    meta.get("summary") or meta.get("description") or ""
                ).strip(),
                "cover": _cover(meta.get("cover") or meta.get("image"), page_url),
            }
        )
    records.sort(key=lambda item: item["date"], reverse=True)
    return records


def _media(item: dict, modifier: str) -> str:
    if item["cover"]:
        source = html.escape(item["cover"], quote=True)
        return (
            f'<span class="{modifier}">'
            f'<img src="{source}" alt="" loading="lazy" />'
            "</span>"
        )
    return f'<span class="{modifier} {modifier}--empty"></span>'


def _meta(item: dict, locale: str) -> str:
    text = " · ".join(
        part for part in (_format_date(item["date"], locale), item["authors"]) if part
    )
    return f'<span class="ucm-blog-meta">{html.escape(text)}</span>'


def _banner(item: dict, page_url: str, locale: str, latest: bool = False) -> str:
    href = get_relative_url(f"{page_url}{item['slug']}/", page_url)
    badge = (
        f'<span class="ucm-blog-badge">{LABELS[locale]["latest"]}</span>'
        if latest
        else ""
    )
    return (
        f'<a class="ucm-blog-banner" href="{href}">'
        f"{_media(item, 'ucm-blog-banner__media')}"
        '<span class="ucm-blog-banner__body">'
        f"{badge}"
        f'<span class="ucm-blog-banner__title">{html.escape(item["title"])}</span>'
        f"{_meta(item, locale)}"
        f'<span class="ucm-blog-banner__summary">{html.escape(item["summary"])}</span>'
        "</span></a>"
    )


def _month_label(value: date, locale: str) -> str:
    if locale == "zh":
        return f"{value.month}月"
    return value.strftime("%b")


def _by_year_month(posts: list[dict]) -> list[tuple[int, list[tuple[int, list[dict]]]]]:
    groups: list[tuple[int, list[tuple[int, list[dict]]]]] = []
    for post in posts:
        year = post["date"].year
        month = post["date"].month
        if not groups or groups[-1][0] != year:
            groups.append((year, []))
        months = groups[-1][1]
        if not months or months[-1][0] != month:
            months.append((month, []))
        months[-1][1].append(post)
    return groups


def _render(page) -> str:
    locale = _locale(page.file.src_uri)
    posts = _records(Path(page.file.abs_src_path).parent, page.url)
    if not posts:
        return f'<p class="ucm-blog-empty">{LABELS[locale]["empty"]}</p>'

    groups = _by_year_month(posts)
    chunks = ['<div class="ucm-blog">']
    latest_done = False
    for year, months in groups:
        chunks.append(
            f'<section class="ucm-blog-year-block" id="year-{year}"'
            f' data-year="{year}">'
        )
        chunks.append(f'<p class="ucm-blog-year">{year}</p>')
        for month, items in months:
            chunks.append(f'<div class="ucm-blog-month" id="month-{year}-{month:02d}">')
            for item in items:
                latest = not latest_done
                latest_done = True
                chunks.append(_banner(item, page.url, locale, latest))
            chunks.append("</div>")
        chunks.append("</section>")
    chunks.append("</div>")
    return "".join(chunks)


def _post_meta(page) -> str:
    meta = page.meta or {}
    if not meta.get("date"):
        return ""
    locale = _locale(page.file.src_uri)
    published = _format_date(_date(meta["date"], Path(page.file.abs_src_path)), locale)
    text = " · ".join(part for part in (published, _authors(meta)) if part)
    return f'<p class="ucm-blog-post-meta">{html.escape(text)}</p>'


def _back_link(page) -> str:
    locale = _locale(page.file.src_uri)
    return (
        f'<a class="ucm-blog-back" href="../" data-blog-back>'
        '<span class="ucm-blog-back__icon" aria-hidden="true">←</span>'
        f'<span class="ucm-blog-back__label">{LABELS[locale]["back"]}</span>'
        "</a>"
    )


def on_page_content(html_text, page, config, files):
    source = page.file.src_uri
    if "/blog/" not in f"/{source}":
        return html_text
    if source.endswith("blog/index.md"):
        for pattern in (LIST_MARKER, f"<p>{LIST_MARKER}</p>"):
            if pattern in html_text:
                return html_text.replace(pattern, _render(page), 1)
        LOG.warning(
            f"Blog index '{source}' is missing the '{LIST_MARKER}' marker; "
            "the post list was not rendered."
        )
        return html_text

    back = _back_link(page)
    if "<h1" in html_text:
        html_text = html_text.replace("<h1", f"{back}<h1", 1)
    else:
        html_text = back + html_text

    meta = _post_meta(page)
    if meta and "</h1>" in html_text:
        html_text = html_text.replace("</h1>", f"</h1>\n{meta}", 1)
    return html_text


def _timeline(page) -> dict:
    locale = _locale(page.file.src_uri)
    is_index = page.file.src_uri.endswith("index.md")
    if is_index:
        base = page.url
        current = None
    else:
        base = page.url[: -(len(page.file.name) + 1)]
        meta, _ = _parse(Path(page.file.abs_src_path))
        current = _date(meta.get("date"), Path(page.file.abs_src_path))

    posts = _records(Path(page.file.abs_src_path).parent, page.url)
    years = []
    for post in posts:
        year = post["date"].year
        month = post["date"].month
        if not years or years[-1]["year"] != year:
            years.append({"year": year, "count": 0, "months": []})
        entry = years[-1]
        entry["count"] += 1
        if not entry["months"] or entry["months"][-1]["month"] != month:
            entry["months"].append(
                {
                    "month": month,
                    "count": 0,
                    "label": _month_label(post["date"], locale),
                }
            )
        entry["months"][-1]["count"] += 1

    for entry in years:
        entry["active"] = bool(current and current.year == entry["year"])
        for month in entry["months"]:
            month["active"] = bool(
                current
                and current.year == entry["year"]
                and current.month == month["month"]
            )

    return {
        "home": base,
        "nav_label": "打开主导航" if locale == "zh" else "Open main navigation",
        "years": years,
    }


def on_page_context(context, page, config, nav):
    if "/blog/" in f"/{page.file.src_uri}":
        context["ucm_blog_timeline"] = _timeline(page)
    return context
