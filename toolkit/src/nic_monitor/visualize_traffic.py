#!/usr/bin/env python3
"""网络流量可视化工具 — 解析 net_monitor_pro.sh 生成的 CSV 数据并绘图。"""

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 解析人类可读的速度/容量字符串 ──────────────────────────────────

UNIT_SCALE_BPS = {"B/s": 1, "KB/s": 1024, "MB/s": 1024**2, "GB/s": 1024**3}
UNIT_SCALE_BYTES = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def parse_rate(s: str) -> float:
    """将 '4.34 KB/s' 解析为 bytes/sec 浮点数。"""
    s = s.strip()
    if s == "0.0 B/s" or s == "0 B/s":
        return 0.0
    m = re.match(r"([\d.]+)\s*(B/s|KB/s|MB/s|GB/s)", s)
    if m:
        return float(m.group(1)) * UNIT_SCALE_BPS[m.group(2)]
    return 0.0


def parse_speed(s: str) -> float:
    """将 '200Gb/s' / '25Mb/s' 解析为 Mbps 浮点数。"""
    s = s.strip()
    if s in ("---", "N/A"):
        return 0.0
    m = re.match(r"([\d.]+)\s*(Gb/s|Mb/s)", s)
    if m:
        val = float(m.group(1))
        return val * 1000 if m.group(2) == "Gb/s" else val
    return 0.0


def parse_util(s: str) -> float:
    """将 '0%' / '42%' / 'N/A' 解析为百分比浮点数。"""
    s = s.strip()
    if s == "N/A":
        return -1.0
    m = re.match(r"([\d.]+)%", s)
    return float(m.group(1)) if m else -1.0


# ── 格式化器 ──────────────────────────────────────────────────────


def rate_formatter_bps(x, _):
    if x >= 1024**3:
        return f"{x / 1024**3:.1f} GB/s"
    if x >= 1024**2:
        return f"{x / 1024**2:.1f} MB/s"
    if x >= 1024:
        return f"{x / 1024:.1f} KB/s"
    return f"{x:.0f} B/s"


def rate_formatter_mbps(x, _):
    if x >= 1000:
        return f"{x / 1000:.1f} Gb/s"
    return f"{x:.0f} Mb/s"


# ── 数据加载 ──────────────────────────────────────────────────────


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="utf-8")
    df["接收速率_Bps"] = df["接收速率"].map(parse_rate)
    df["发送速率_Bps"] = df["发送速率"].map(parse_rate)
    df["端口速率_Mbps"] = df["端口速率"].map(parse_speed)
    df["利用率_pct"] = df["利用率"].map(parse_util)
    df["时间"] = pd.to_datetime(df["时间"], format="%Y-%m-%d %H:%M:%S")
    return df


# ── 绘图 ──────────────────────────────────────────────────────────

PALETTE = plt.cm.Set2.colors


def plot_traffic(df: pd.DataFrame, output_dir: Path, interfaces=None):
    """绘制每块网卡的 RX/TX 流量时序图 + 利用率图。"""
    ifs = sorted(df["网卡"].unique())
    if interfaces:
        ifs = [i for i in ifs if i in interfaces]
    n = len(ifs)
    if n == 0:
        print("无匹配网卡数据")
        return

    # ─ 图1: 流量时序（每网卡一行） ─
    fig1, axes1 = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    if n == 1:
        axes1 = [axes1]
    fig1.suptitle("网卡流量时序图", fontsize=14, fontweight="bold", y=0.98)

    for idx, eth in enumerate(ifs):
        ax = axes1[idx]
        sub = df[df["网卡"] == eth].sort_values("时间")
        speed_mbps = sub["端口速率_Mbps"].iloc[0] if len(sub) > 0 else 0

        ax.fill_between(
            sub["时间"],
            sub["接收速率_Bps"],
            alpha=0.3,
            color=PALETTE[0],
            label="接收 (RX)",
        )
        ax.plot(sub["时间"], sub["接收速率_Bps"], color=PALETTE[0], linewidth=1.2)
        ax.fill_between(
            sub["时间"],
            sub["发送速率_Bps"],
            alpha=0.3,
            color=PALETTE[1],
            label="发送 (TX)",
        )
        ax.plot(sub["时间"], sub["发送速率_Bps"], color=PALETTE[1], linewidth=1.2)

        # 端口带宽上限参考线
        if speed_mbps > 0:
            max_bps = speed_mbps * 1000 * 1000 / 8
            ax.axhline(
                max_bps,
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label=f"端口上限 ({rate_formatter_mbps(speed_mbps, None)})",
            )

        ax.set_title(
            f"{eth}  (驱动: {sub['驱动'].iloc[0]}, 端口: {sub['端口速率'].iloc[0]})",
            fontsize=10,
        )
        ax.set_ylabel("速率")
        ax.yaxis.set_major_formatter(FuncFormatter(rate_formatter_bps))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes1[-1].set_xlabel("时间")
    axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes1[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = output_dir / "traffic_timeseries.png"
    fig1.savefig(out1, dpi=150)
    print(f"已保存: {out1}")

    # ─ 图2: 利用率时序 ─
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    fig2.suptitle("网卡利用率时序图", fontsize=14, fontweight="bold")
    for idx, eth in enumerate(ifs):
        sub = df[df["网卡"] == eth].sort_values("时间")
        valid = sub[sub["利用率_pct"] >= 0]
        if len(valid) == 0:
            continue
        ax2.plot(
            valid["时间"],
            valid["利用率_pct"],
            label=eth,
            linewidth=1.5,
            color=PALETTE[idx % len(PALETTE)],
        )

    ax2.axhline(
        80, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="80% 告警线"
    )
    ax2.axhline(
        30, color="orange", linestyle="--", linewidth=0.8, alpha=0.4, label="30% 警戒线"
    )
    ax2.set_ylabel("利用率 (%)")
    ax2.set_ylim(
        0,
        max(
            105,
            (
                df[df["利用率_pct"] >= 0]["利用率_pct"].max() * 1.1
                if len(df[df["利用率_pct"] >= 0]) > 0
                else 105
            ),
        ),
    )
    ax2.set_xlabel("时间")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = output_dir / "utilization_timeseries.png"
    fig2.savefig(out2, dpi=150)
    print(f"已保存: {out2}")

    # ─ 图3: RX+TX 总流量对比（堆叠面积） ─
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    fig3.suptitle("所有网卡总流量对比 (堆叠面积图)", fontsize=14, fontweight="bold")
    # 按时间对齐: 同一时刻的 RX 和 TX 分别堆叠
    pivoted_rx = df.pivot_table(
        index="时间", columns="网卡", values="接收速率_Bps", aggfunc="mean"
    ).fillna(0)
    pivoted_tx = df.pivot_table(
        index="时间", columns="网卡", values="发送速率_Bps", aggfunc="mean"
    ).fillna(0)

    bottom_rx = 0
    for idx, eth in enumerate(ifs):
        if eth not in pivoted_rx.columns:
            continue
        ax3.fill_between(
            pivoted_rx.index,
            bottom_rx,
            bottom_rx + pivoted_rx[eth],
            alpha=0.4,
            color=PALETTE[idx % len(PALETTE)],
            label=f"{eth} RX",
        )
        bottom_rx += pivoted_rx[eth]

    bottom_tx = 0
    for idx, eth in enumerate(ifs):
        if eth not in pivoted_tx.columns:
            continue
        ax3.fill_between(
            pivoted_tx.index,
            bottom_tx,
            bottom_tx + pivoted_tx[eth],
            alpha=0.25,
            color=PALETTE[idx % len(PALETTE)],
            label=f"{eth} TX",
            linestyle="--",
        )
        bottom_tx += pivoted_tx[eth]

    ax3.yaxis.set_major_formatter(FuncFormatter(rate_formatter_bps))
    ax3.set_ylabel("速率")
    ax3.set_xlabel("时间")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.legend(loc="upper right", fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    out3 = output_dir / "total_traffic_stacked.png"
    fig3.savefig(out3, dpi=150)
    print(f"已保存: {out3}")

    # ─ 图4: 网卡流量峰值/均值柱状统计 ─
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))
    fig4.suptitle("网卡流量统计摘要", fontsize=14, fontweight="bold")

    stats_rx = []
    stats_tx = []
    labels = []
    for eth in ifs:
        sub = df[df["网卡"] == eth]
        labels.append(eth)
        stats_rx.append(
            {"峰值": sub["接收速率_Bps"].max(), "均值": sub["接收速率_Bps"].mean()}
        )
        stats_tx.append(
            {"峰值": sub["发送速率_Bps"].max(), "均值": sub["发送速率_Bps"].mean()}
        )

    x = range(len(labels))
    width = 0.35

    for ax_side, stats, title, color_idx in [
        (axes4[0], stats_rx, "接收 (RX)", 0),
        (axes4[1], stats_tx, "发送 (TX)", 1),
    ]:
        peak_vals = [s["峰值"] for s in stats]
        avg_vals = [s["均值"] for s in stats]
        ax_side.bar(
            [i - width / 2 for i in x],
            peak_vals,
            width,
            label="峰值",
            color=PALETTE[color_idx],
            alpha=0.7,
        )
        ax_side.bar(
            [i + width / 2 for i in x],
            avg_vals,
            width,
            label="均值",
            color=PALETTE[color_idx + 2],
            alpha=0.7,
        )
        ax_side.set_xticks(x)
        ax_side.set_xticklabels(labels, fontsize=9)
        ax_side.set_title(title)
        ax_side.yaxis.set_major_formatter(FuncFormatter(rate_formatter_bps))
        ax_side.legend(fontsize=8)
        ax_side.grid(True, alpha=0.3, axis="y")

    fig4.tight_layout()
    out4 = output_dir / "traffic_stats_summary.png"
    fig4.savefig(out4, dpi=150)
    print(f"已保存: {out4}")

    plt.close("all")


# ── 主入口 ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="解析 net_monitor_pro.sh 的 CSV 日志，生成网络流量可视化图表"
    )
    parser.add_argument(
        "csv", nargs="*", help="CSV 文件路径（不指定则自动查找当前目录下 *.csv）"
    )
    parser.add_argument(
        "-o", "--output", default=".", help="图片输出目录（默认当前目录）"
    )
    parser.add_argument(
        "-i", "--interfaces", nargs="*", help="只绘制指定网卡（如 enp194s0f0）"
    )
    args = parser.parse_args()

    csv_files = args.csv or sorted(glob.glob("*.csv"))
    if not csv_files:
        print("未找到 CSV 文件，请指定路径或将 CSV 放在当前目录下。")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        print(f"\n处理: {csv_file}")
        df = load_csv(csv_file)
        if df.empty:
            print(f"  {csv_file} 数据为空，跳过。")
            continue
        # 输出目录按 CSV 文件名分组
        base = Path(csv_file).stem
        sub_dir = output_dir / base
        sub_dir.mkdir(parents=True, exist_ok=True)
        plot_traffic(df, sub_dir, args.interfaces)

    print("\n全部图表生成完毕。")


if __name__ == "__main__":
    main()
