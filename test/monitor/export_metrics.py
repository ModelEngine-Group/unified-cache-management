# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VictoriaMetrics Metrics Exporter and Visualizer.

Automatically exports all metrics from VictoriaMetrics instance and generates
time-series visualizations. Compatible with vLLM/sglang metrics endpoints.

Usage:
    python export_metrics.py --vm-url http://localhost:8428 --duration 2h --output ./metrics_export
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import matplotlib
import pandas as pd
import requests

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Constants
DEFAULT_VM_URL = "http://localhost:8428"
DEFAULT_DURATION_HOURS = 1
DEFAULT_STEP_SECONDS = 10
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
SKIP_METRIC_PREFIXES = ("scrape_", "up", "vm_", "go_", "process_")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class VMExporter:
    """VictoriaMetrics data exporter and visualizer."""

    def __init__(
        self, vm_url: str, output_dir: Path, duration_hours: int, step_seconds: int
    ):
        """
        Initialize exporter.

        Args:
            vm_url: VictoriaMetrics HTTP API endpoint
            output_dir: Directory for output files
            duration_hours: Time window for data export
            step_seconds: Sampling interval in seconds
        """
        self.vm_url = vm_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.duration_hours = duration_hours
        self.step_seconds = step_seconds
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        # Calculate time range
        self.end_ts = int(datetime.now().timestamp())
        self.start_ts = self.end_ts - (duration_hours * 3600)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_metric_names(self) -> List[str]:
        """
        Fetch all metric names from VM.

        Returns:
            List of metric name strings

        Raises:
            requests.RequestException: If API call fails
        """
        url = urljoin(self.vm_url, "/api/v1/label/__name__/values")

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=DEFAULT_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") != "success":
                    raise ValueError(f"API returned non-success status: {data}")

                all_metrics = data.get("data", [])

                # Filter out internal/system metrics
                filtered = [
                    m
                    for m in all_metrics
                    if not any(m.startswith(p) for p in SKIP_METRIC_PREFIXES)
                ]

                logger.info(
                    f"Discovered {len(all_metrics)} metrics, {len(filtered)} after filtering"
                )
                return filtered

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise

        return []

    def query_range(self, metric: str) -> Optional[List[Dict]]:
        """
        Query time-series data for a specific metric.

        Args:
            metric: Metric name to query

        Returns:
            List of time-series results or None if query fails
        """
        url = urljoin(self.vm_url, "/api/v1/query_range")
        params = {
            "query": metric,
            "start": self.start_ts,
            "end": self.end_ts,
            "step": f"{self.step_seconds}s",
        }

        try:
            resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "success":
                return None

            return data.get("data", {}).get("result", [])

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.debug(f"Query failed for {metric}: {e}")
            return None

    def parse_series_to_dataframe(
        self, metric: str, result: List[Dict]
    ) -> Optional[pd.DataFrame]:
        """
        Convert VM query result to pandas DataFrame.

        Args:
            metric: Metric name
            result: Raw API result list

        Returns:
            DataFrame with columns: timestamp, metric, value, [labels...]
        """
        if not result:
            return None

        rows = []
        for series in result:
            metric_labels = series.get("metric", {})
            values = series.get("values", [])

            for timestamp, value in values:
                try:
                    row = {
                        "timestamp": datetime.fromtimestamp(timestamp),
                        "metric": metric,
                        "value": float(value),
                    }
                    # Add all labels as columns (excluding __name__ which is redundant)
                    for label_key, label_val in metric_labels.items():
                        if label_key != "__name__":
                            row[label_key] = label_val
                    rows.append(row)
                except (ValueError, TypeError):
                    continue

        if not rows:
            return None

        return pd.DataFrame(rows)

    def export_to_csv(self, metric: str, df: pd.DataFrame) -> Path:
        """
        Export DataFrame to CSV file.

        Args:
            metric: Metric name (used for filename)
            df: DataFrame to export

        Returns:
            Path to exported file
        """
        # Sanitize filename (replace colons and special chars)
        safe_name = metric.replace(":", "_").replace("/", "_")
        filepath = self.output_dir / f"{safe_name}.csv"

        df.to_csv(filepath, index=False, float_format="%.6f")
        logger.info(f"Exported CSV: {filepath.name} ({len(df)} rows)")
        return filepath

    def plot_timeseries(self, metric: str, df: pd.DataFrame) -> Optional[Path]:
        """
        Generate time-series plot for metric.

        Args:
            metric: Metric name
            df: DataFrame with timestamp and value columns

        Returns:
            Path to generated PNG file or None if skipped
        """
        # Skip if too many unique series (would clutter the plot)
        series_count = df.groupby(
            [c for c in df.columns if c not in ["timestamp", "value", "metric"]]
        ).ngroups

        if series_count > 20:
            logger.debug(
                f"Skipping plot for {metric}: too many series ({series_count})"
            )
            return None

        try:
            plt.figure(figsize=(12, 6))

            # Group by labels to plot separate lines
            label_cols = [
                c for c in df.columns if c not in ["timestamp", "value", "metric"]
            ]

            if label_cols:
                grouped = df.groupby(label_cols)
                for group_keys, group_df in grouped:
                    label_str = ", ".join(
                        f"{k}={v}"
                        for k, v in zip(label_cols, group_keys)
                        if k != "__name__"
                    )
                    if len(label_str) > 50:
                        label_str = label_str[:47] + "..."

                    plt.plot(
                        group_df["timestamp"],
                        group_df["value"],
                        marker="o",
                        markersize=2,
                        linewidth=1,
                        label=label_str,
                    )
                plt.legend(loc="best", fontsize=8, framealpha=0.9)
            else:
                plt.plot(
                    df["timestamp"],
                    df["value"],
                    marker="o",
                    markersize=2,
                    linewidth=1.5,
                    color="#2E86AB",
                )

            plt.title(
                f"{metric} (Last {self.duration_hours}h)",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.grid(True, alpha=0.3, linestyle="--")
            plt.xticks(rotation=45, ha="right")

            # Format x-axis
            ax = plt.gca()
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

            plt.tight_layout()

            safe_name = metric.replace(":", "_").replace("/", "_")
            filepath = self.output_dir / f"{safe_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Generated plot: {filepath.name}")
            return filepath

        except Exception as e:
            logger.warning(f"Plot generation failed for {metric}: {e}")
            plt.close()
            return None

    def generate_summary_report(self, metrics: List[str], exported_count: int):
        """Generate markdown summary report."""
        report_path = self.output_dir / "_summary.md"

        time_range_str = f"{datetime.fromtimestamp(self.start_ts)} ~ {datetime.fromtimestamp(self.end_ts)}"

        content = f"""# VM Metrics Export Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Source:** {self.vm_url}  
**Time Range:** {time_range_str}  
**Total Metrics:** {len(metrics)}  
**Successfully Exported:** {exported_count}

## Metrics List

| Metric | Description |
|--------|-------------|
{chr(10).join(f"| {m} | Auto-exported from VM |" for m in sorted(metrics))}

## File Structure

- `*.csv`: Raw time-series data (one file per metric)
- `*.png`: Visualization plots (generated for metrics with < 20 series)
- `_summary.md`: This report

## Quick Analysis

View CSV files with:
```bash
# Example: View first 10 rows of a metric
head -n 10 {metrics[0].replace(':', '_') if metrics else 'metric_name'}.csv
```

Or import into Excel/Pandas for further analysis.
"""

        report_path.write_text(content, encoding="utf-8")
        logger.info(f"Summary report: {report_path}")

    def run(self):
        """Execute full export workflow."""
        logger.info(f"Starting export from {self.vm_url}")
        logger.info(
            f"Time range: {datetime.fromtimestamp(self.start_ts)} to {datetime.fromtimestamp(self.end_ts)}"
        )

        # Step 1: Get metric list
        try:
            metrics = self.get_metric_names()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch metric list: {e}")
            sys.exit(1)

        if not metrics:
            logger.warning("No metrics found or all filtered out")
            sys.exit(0)

        # Step 2: Export each metric
        exported_count = 0
        failed_metrics = []

        for idx, metric in enumerate(metrics, 1):
            logger.info(f"[{idx}/{len(metrics)}] Processing: {metric}")

            # Query data
            result = self.query_range(metric)
            if not result:
                failed_metrics.append(metric)
                continue

            # Convert to DataFrame
            df = self.parse_series_to_dataframe(metric, result)
            if df is None or df.empty:
                logger.warning(f"No data points for {metric}")
                continue

            # Export CSV
            self.export_to_csv(metric, df)

            # Generate plot
            self.plot_timeseries(metric, df)

            exported_count += 1

        # Step 3: Generate report
        self.generate_summary_report(metrics, exported_count)

        # Final status
        logger.info("=" * 50)
        logger.info(f"Export complete: {self.output_dir.absolute()}")
        logger.info(f"Total metrics: {len(metrics)}")
        logger.info(f"Successfully exported: {exported_count}")
        logger.info(f"Failed: {len(failed_metrics)}")
        if failed_metrics:
            logger.debug(f"Failed metrics: {failed_metrics}")


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to hours.
    Supports: 1h, 30m, 1d

    Args:
        duration_str: Duration string

    Returns:
        Hours as integer

    Raises:
        ValueError: If format invalid
    """
    if not duration_str:
        return DEFAULT_DURATION_HOURS

    unit = duration_str[-1].lower()
    try:
        value = int(duration_str[:-1])
    except ValueError:
        raise ValueError(f"Invalid duration format: {duration_str}")

    if unit == "h":
        return value
    elif unit == "m":
        return max(1, value // 60)
    elif unit == "d":
        return value * 24
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export VictoriaMetrics data and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export last hour with default settings
  python export_metrics.py --vm-url http://localhost:8428

  # Export 2 hours to specific directory
  python export_metrics.py --vm-url http://vm-host:8428 --duration 2h --output ./export

  # Export with 5s sampling interval
  python export_metrics.py --vm-url http://vm-host:8428 --step 5 --duration 30m
        """,
    )

    parser.add_argument(
        "--vm-url",
        default=DEFAULT_VM_URL,
        help=f"VictoriaMetrics HTTP API URL (default: {DEFAULT_VM_URL})",
    )
    parser.add_argument(
        "--duration",
        default=f"{DEFAULT_DURATION_HOURS}h",
        help="Time window to export (e.g., 1h, 30m, 1d). Default: 1h",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP_SECONDS,
        help=f"Sampling interval in seconds (default: {DEFAULT_STEP_SECONDS})",
    )
    parser.add_argument(
        "--output",
        default="./vm_export",
        help="Output directory for CSV and PNG files (default: ./vm_export)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        duration_hours = parse_duration(args.duration)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    exporter = VMExporter(
        vm_url=args.vm_url,
        output_dir=Path(args.output),
        duration_hours=duration_hours,
        step_seconds=args.step,
    )

    exporter.run()


if __name__ == "__main__":
    main()
