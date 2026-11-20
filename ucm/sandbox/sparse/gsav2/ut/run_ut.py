# -*- coding: utf-8 -*-
"""一键运行 gsav2 UT 并分别统计行覆盖率 / 分支覆盖率。

用法（在 ut/ 目录内执行）：
    python3 -X utf8 run_ut.py          # Linux / NPU 服务器
    py -3 -X utf8 run_ut.py            # Windows

效果：
    1. 运行完整测试套件，全部输出保存到 ut_result.log；
    2. 终端额外打印一张表，行覆盖率与分支覆盖率**分开统计**
       （pytest 自带的 Cover% 是两者合并后的综合值）；
    3. 退出码沿用 pytest（覆盖率低于 90% 门槛时非零，可用于 CI）。
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "ut_result.log"
COV_JSON = HERE / "cov.json"


def main() -> int:
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "pytest",
        "-q",
        f"--cov-report=json:{COV_JSON}",
    ]
    with open(LOG, "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT)
    print(f"[run_ut] 完整日志已保存: {LOG}\n")

    data = json.loads(COV_JSON.read_text(encoding="utf-8"))
    rows = []
    for name, info in data["files"].items():
        s = info["summary"]
        rows.append(
            (
                Path(name).name,
                s["covered_lines"],
                s["num_statements"],
                s["covered_branches"],
                s["num_branches"],
            )
        )
    rows.sort()

    t = data["totals"]
    rows.append(
        (
            "TOTAL",
            t["covered_lines"],
            t["num_statements"],
            t["covered_branches"],
            t["num_branches"],
        )
    )

    def pct(a, b):
        return "  n/a" if b == 0 else f"{100 * a / b:6.2f}%"

    name_w = max(len(r[0]) for r in rows)
    print(f"{'模块':<{name_w}}  {'行覆盖率':>14}  {'分支覆盖率':>16}")
    print(f"{'':<{name_w}}  {'(cov/total)':>14}  {'(cov/total)':>16}")
    print("-" * (name_w + 36))
    for name, lc, lt, bc, bt in rows:
        line = f"{name:<{name_w}}  {pct(lc, lt)} ({lc}/{lt})"
        line += f"  {pct(bc, bt)} ({bc}/{bt})"
        print(line)
    print("-" * (name_w + 36))
    ok_line = t["covered_lines"] == t["num_statements"]
    ok_branch = (
        t["num_branches"] > 0 and 100 * t["covered_branches"] / t["num_branches"] > 90
    )
    print(
        f"行覆盖 100%: {'达标' if ok_line else '未达标'} | "
        f"分支覆盖 >90%: {'达标' if ok_branch else '未达标'}"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
