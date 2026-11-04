import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from scripts import collect_ucm_metrics as metrics


def _histogram_text(
    rank: str,
    buckets: dict[str, float],
    count: float,
    total: float,
    reverse_labels: bool = False,
) -> str:
    labels = [
        ("model_name", "model-a"),
        ("engine", "engine-0"),
        ("worker_rank", rank),
    ]
    if reverse_labels:
        labels.reverse()

    def label_text(extra: tuple[str, str] | None = None) -> str:
        items = [*labels, *([extra] if extra else [])]
        return ",".join(f'{key}="{value}"' for key, value in items)

    lines = [
        f'ucm:cache_load_duration_ms_bucket{{{label_text(("le", le))}}} {value}'
        for le, value in buckets.items()
    ]
    lines.extend(
        (
            f"ucm:cache_load_duration_ms_count{{{label_text()}}} {count}",
            f"ucm:cache_load_duration_ms_sum{{{label_text()}}} {total}",
        )
    )
    return "\n".join(lines)


def _snapshots():
    before_text = "\n".join(
        (
            _histogram_text("0", {"10": 1, "20": 1, "+Inf": 1}, 1, 5),
            _histogram_text("1", {"10": 0, "20": 2, "+Inf": 2}, 2, 30),
        )
    )
    after_text = "\n".join(
        (
            _histogram_text(
                "0",
                {"10": 2, "20": 3, "+Inf": 3},
                3,
                25,
                reverse_labels=True,
            ),
            _histogram_text(
                "1",
                {"10": 0, "20": 4, "+Inf": 4},
                4,
                60,
                reverse_labels=True,
            ),
        )
    )
    return metrics.parse_samples(before_text), metrics.parse_samples(after_text)


class CollectUcmMetricsTest(unittest.TestCase):
    def test_histogram_stats_keep_ranks_separate_and_merge_all_rank_buckets(self):
        before, after = _snapshots()

        per_rank = metrics.histogram_intervals_by_rank(
            "cache_load_duration_ms", before, after
        )
        rank0 = next(
            stats for group, stats in per_rank.items() if group.worker_rank == "0"
        )
        rank1 = next(
            stats for group, stats in per_rank.items() if group.worker_rank == "1"
        )

        self.assertEqual(rank0.count, 2)
        self.assertEqual(rank0.average, 10)
        self.assertAlmostEqual(rank0.quantiles[0.9], 18)
        self.assertAlmostEqual(rank0.quantiles[0.99], 19.8)
        self.assertEqual(rank1.count, 2)
        self.assertEqual(rank1.average, 15)
        self.assertAlmostEqual(rank1.quantiles[0.9], 19)
        self.assertAlmostEqual(rank1.quantiles[0.99], 19.9)

        all_ranks = metrics.histogram_intervals_by_rank(
            "cache_load_duration_ms", before, after, aggregate_ranks=True
        )
        self.assertEqual(len(all_ranks), 1)
        group, combined = next(iter(all_ranks.items()))
        self.assertEqual(group.worker_rank, "ALL")
        self.assertEqual(combined.count, 4)
        self.assertEqual(combined.average, 12.5)
        self.assertAlmostEqual(combined.quantiles[0.9], 18.6666666667)
        self.assertAlmostEqual(combined.quantiles[0.99], 19.8666666667)
        rank_p90_average = (rank0.quantiles[0.9] + rank1.quantiles[0.9]) / 2
        self.assertNotAlmostEqual(combined.quantiles[0.9], rank_p90_average)

    def test_report_has_three_pages_with_all_ranks_before_rank_detail(self):
        before, after = _snapshots()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            report_path = output_dir / "metrics_report.md"

            summary = metrics.write_delta_report(
                report_path,
                before,
                after,
                window_seconds=10,
                full=False,
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertLess(report.index("## Connector"), report.index("## CacheStore"))
            self.assertLess(report.index("## CacheStore"), report.index("## Posix"))
            cache_page = report[
                report.index("## CacheStore") : report.index("## Posix")
            ]
            self.assertLess(
                cache_page.index("### All ranks"),
                cache_page.index("### Per-rank detail"),
            )
            self.assertIn(
                "| model-a | engine-0 | ALL | cache_load_duration_ms | 4 | 12.500",
                cache_page,
            )
            self.assertIn(
                "| model-a | engine-0 | 0 | cache_load_duration_ms | 2 | 10",
                cache_page,
            )
            self.assertIn(
                "| model-a | engine-0 | 1 | cache_load_duration_ms | 2 | 15",
                cache_page,
            )

            cache_summary = summary["pages"]["cachestore"]
            self.assertEqual(cache_summary["all_ranks"][0]["worker_rank"], "ALL")
            self.assertEqual(len(cache_summary["per_rank"]), 2)

            csv_path = output_dir / "metrics_report.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            cache_rows = [row for row in rows if row["page"] == "cachestore"]
            self.assertEqual(cache_rows[0]["scope"], "all_ranks")
            self.assertEqual(cache_rows[0]["worker_rank"], "ALL")
            self.assertEqual([row["worker_rank"] for row in cache_rows[1:]], ["0", "1"])

            xlsx_path = output_dir / "metrics_report.xlsx"
            self.assertTrue(xlsx_path.exists())
            with zipfile.ZipFile(xlsx_path) as workbook:
                workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
                namespace = {
                    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                }
                sheet_names = [
                    sheet.attrib["name"]
                    for sheet in workbook_xml.findall(
                        "main:sheets/main:sheet", namespace
                    )
                ]
                self.assertEqual(sheet_names, ["Connector", "CacheStore", "Posix"])
                cache_sheet = workbook.read("xl/worksheets/sheet2.xml").decode()
                self.assertLess(
                    cache_sheet.index("All ranks"),
                    cache_sheet.index("Per-rank detail"),
                )
                self.assertIn("cache_load_duration_ms", cache_sheet)

    def test_report_stem_names_csv_and_xlsx(self):
        before, after = _snapshots()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics.write_delta_report(
                output_dir / "metrics_report.md",
                before,
                after,
                window_seconds=10,
                full=False,
                report_stem="metrics_report_posix-hit_20261012-153000",
            )

            self.assertTrue(
                (output_dir / "metrics_report_posix-hit_20261012-153000.csv").exists()
            )
            self.assertTrue(
                (output_dir / "metrics_report_posix-hit_20261012-153000.xlsx").exists()
            )


if __name__ == "__main__":
    unittest.main()
