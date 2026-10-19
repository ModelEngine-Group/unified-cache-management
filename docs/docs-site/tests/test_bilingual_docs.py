"""Check the complete bilingual documentation tree without Git history."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_bilingual_docs import missing_pairs


@pytest.mark.parametrize("language,other", [("en", "zh"), ("zh", "en")])
def test_every_markdown_path_requires_its_counterpart(tmp_path, language, other):
    page = tmp_path / language / "nested/page.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Page")
    assert missing_pairs(tmp_path) == [
        (f"{language}/nested/page.md", f"{other}/nested/page.md")
    ]
    counterpart = tmp_path / other / "nested/page.md"
    counterpart.parent.mkdir(parents=True)
    counterpart.write_text("# 对应页面")
    (page.parent / "image.svg").write_text("<svg/>")
    assert missing_pairs(tmp_path) == []
    counterpart.unlink()
    assert missing_pairs(tmp_path) == [
        (f"{language}/nested/page.md", f"{other}/nested/page.md")
    ]
