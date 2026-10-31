#!/usr/bin/env bash
# ============================================================================
# gsa_prefetch 测试一键脚本：编译 → 运行 → 覆盖率 + 用例统计报告
#
# 功能：
#   1. 自动探测 torch / python / gtest / OpenMP 依赖路径
#   2. 将 gtest 源码编译为静态库（一次）
#   3. 将 4 个生产源码编译为带覆盖率插桩的 .o（一次）
#   4. 每个测试 .cpp 单独编译并链接成独立可执行文件
#   5. 逐个运行，统计用例总数 / 通过数 / 失败数
#   6. 用 gcov 统计生产源码的行覆盖率 + 分支覆盖率
#   7. 汇总打印报告
#
# 用法：
#   ./run_tests.sh            # 全量编译运行
#   ./run_tests.sh --clean     # 清理后重新编译运行
#   ./run_tests.sh --no-cov    # 不做覆盖率统计（更快）
#   ./run_tests.sh -v          # verbose 输出每个用例详情
# ============================================================================
set -uo pipefail

# ── 颜色 ─────────────────────────────────────────────────────────────────────
RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
BLUE=$'\033[34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
info()  { printf "${BLUE}[INFO]${NC}  %s\n"  "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n"  "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n"  "$*"; }
err()   { printf "${RED}[ERROR]${NC} %s\n" "$*"; }

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFETCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# prefetch/ is 5 levels deep from the repo root (repo/ucm/sandbox/sparse/gsav2/prefetch),
# so the repo root (which contains ucm/ and toolkit/) is 5 levels up.
REPO_ROOT="$(cd "$PREFETCH_DIR/../../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/opencode/prefetch_build}"

# ── 参数解析 ─────────────────────────────────────────────────────────────────
CLEAN=0; DO_COV=1; VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --clean)   CLEAN=1 ;;
        --no-cov)  DO_COV=0 ;;
        -v|--verbose) VERBOSE=1 ;;
        *) err "未知参数: $arg"; exit 2 ;;
    esac
done

# ── 清理 ─────────────────────────────────────────────────────────────────────
if [[ $CLEAN -eq 1 ]]; then
    info "清理构建目录 $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR/objs" "$BUILD_DIR/bin"

# ============================================================================
# 1. 依赖探测
# ============================================================================
info "探测依赖..."

# Python (torch 内部依赖 Python.h)
PYTHON_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))' 2>/dev/null)"
PYTHON_LIB_DIR="$(python3 -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))' 2>/dev/null)"
PYTHON_VER="$(python3 -c 'import sysconfig;print(f"python{sysconfig.get_python_version()}")' 2>/dev/null)"
[[ -z "$PYTHON_INC" || -z "$PYTHON_LIB_DIR" ]] && { err "找不到 Python 开发头文件"; exit 1; }

# PyTorch
TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
[[ -z "$TORCH_DIR" ]] && { err "找不到 PyTorch"; exit 1; }
TORCH_INC="$TORCH_DIR/include"
TORCH_LIB="$TORCH_DIR/lib"

# GTest 源码（catlass 自带）
GTEST_ROOT=""
for cand in \
    "/vllm-workspace/vllm-ascend/csrc/third_party/catlass/3rdparty/googletest/googletest" \
    "$REPO_ROOT/toolkit/src/dev-sandbox/vendor/fmt/test/gtest"; do
    if [[ -f "$cand/src/gtest-all.cc" && -f "$cand/include/gtest/gtest.h" ]]; then
        GTEST_ROOT="$cand"; break
    fi
done
if [[ -z "$GTEST_ROOT" ]]; then
    err "找不到 gtest 源码；请安装 libgtest-dev 或指定 GTEST_ROOT"
    exit 1
fi
GTEST_INC="$GTEST_ROOT/include"
GTEST_SRC="$GTEST_ROOT/src"

# UCM store / shared 头文件
UCM_STORE="$REPO_ROOT/ucm/store"
UCM_SHARED="$REPO_ROOT/ucm/shared/infra"

# 生产源码（kvcache_hbm_manger.h 的覆盖率由测试 TU test_hbm_manager.cpp
# 在 --coverage 插桩下驱动；下方 gcov 步骤会额外对该测试 TU 运行 gcov）
PROD_SOURCES=(
    "$PREFETCH_DIR/src/kvcache_pre.cpp"
    "$PREFETCH_DIR/src/kvcache_pre_scatter.cpp"
    "$PREFETCH_DIR/src/kvcache_task_manger.cpp"
    "$PREFETCH_DIR/ascend/kvcache_trans_ascend.cpp"
)

# 测试源码
TEST_SOURCES=()
while IFS= read -r f; do TEST_SOURCES+=("$f"); done < <(
    find "$SCRIPT_DIR/ut" "$SCRIPT_DIR/mst" "$SCRIPT_DIR/dt" -name '*.cpp' 2>/dev/null | sort)

ok "依赖探测完成"
info "  PyTorch   : $TORCH_DIR"
info "  Python    : $PYTHON_INC"
info "  GTest     : $GTEST_ROOT"
info "  生产源码  : ${#PROD_SOURCES[@]} 个"
info "  测试源码  : ${#TEST_SOURCES[@]} 个"

# ============================================================================
# 2. 公共编译 / 链接参数
# ============================================================================
COV_FLAGS=""
if [[ $DO_COV -eq 1 ]]; then
    COV_FLAGS="--coverage -fprofile-arcs -ftest-coverage"
fi

# fmt: torch 自带的 fmt 头文件以 FMT_API 导出 vformat（需要链接 libfmt）。
# 定义 FMT_HEADER_ONLY 使其改为 inline（无需链接 libfmt）。
FMT_FLAG="-DFMT_HEADER_ONLY"

# ── 配置变更检测：覆盖率开关变化时强制全量重编（PCH / .o 都需重新插桩）──
CONFIG_STAMP="$BUILD_DIR/config.txt"
CURRENT_CONFIG="cov=${DO_COV}"
if [[ -f "$CONFIG_STAMP" ]]; then
    PREV_CONFIG="$(cat "$CONFIG_STAMP" 2>/dev/null)"
else
    PREV_CONFIG=""
fi
if [[ "$PREV_CONFIG" != "$CURRENT_CONFIG" ]]; then
    info "配置变更 ($PREV_CONFIG -> $CURRENT_CONFIG)，强制全量重编"
    rm -f "$BUILD_DIR/pch.h.gch" "$BUILD_DIR"/objs/*.o
    echo "$CURRENT_CONFIG" > "$CONFIG_STAMP"
fi

INCLUDES=(
    -I"$PREFETCH_DIR/include"
    -I"$PREFETCH_DIR/ascend"
    -I"$SCRIPT_DIR/common"
    -I"$UCM_STORE"
    -I"$UCM_STORE/detail"
    -I"$UCM_STORE/detail/type"
    -I"$UCM_SHARED"
    -I"$TORCH_INC"
    -I"$TORCH_INC/torch/csrc/api/include"
    -I"$PYTHON_INC"
    -I"$GTEST_INC"
)

CXX_FLAGS=(
    -std=c++17 -fopenmp $COV_FLAGS $FMT_FLAG
    -DKV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC
    "${INCLUDES[@]}"
    -include "$SCRIPT_DIR/common/pch.h"
    -Wno-unused-parameter -Wno-unused-variable
    -Wno-deprecated-declarations -Wno-deprecated -w
)

LINK_FLAGS=(
    $COV_FLAGS -fopenmp
    -L"$TORCH_LIB" -ltorch -lc10 -ltorch_cpu
    -L"$PYTHON_LIB_DIR" -l"$PYTHON_VER"
    -lpthread -ldl -lrt
)
# -lgcov is appended AFTER the objects (see link invocation) so the linker can
# resolve __gcov_dump (referenced by KVPackAbort's coverage-flush call) even in
# --no-cov mode, where the symbol is not auto-pulled in by the objects.
GCOV_LINK_FLAG="-lgcov"

# ============================================================================
# 2.6. 增量编译辅助：源比 .o 新（或 .o 不存在）才重编
# ============================================================================
needs_rebuild() {
    # $1=source $2=output ; 返回 0 表示需要重编
    [[ ! -f "$2" ]] && return 0
    [[ "$1" -nt "$2" ]] && return 0
    return 1
}

# ============================================================================
# 2.7. 编译预编译头 (PCH) —— 一次解析 torch/pybind11 重型头，后续编译秒级完成
# ============================================================================
info "编译预编译头 (PCH)..."
PCH_FILE="$BUILD_DIR/pch.h.gch"
PCH_SRC="$SCRIPT_DIR/common/pch.h"
if needs_rebuild "$PCH_SRC" "$PCH_FILE"; then
    if ! g++ -std=c++17 -fopenmp $COV_FLAGS $FMT_FLAG -DKV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC \
            "${INCLUDES[@]}" -w \
            -x c++-header "$PCH_SRC" -o "$PCH_FILE" 2>"$BUILD_DIR/pch.err"; then
        err "PCH 编译失败"; cat "$BUILD_DIR/pch.err"; exit 1
    fi
    ok "PCH 就绪: $PCH_FILE"
else
    ok "PCH 已缓存，跳过"
fi

# ============================================================================
# 3. 编译 gtest 静态库（一次）
# ============================================================================
info "编译 gtest 静态库..."
# gtest 不依赖 torch/pybind11，跳过 PCH 并只带 gtest 自身头文件即可。
# gtest-all.cc 内部 #include "src/..." 相对于 googletest 根目录，故 cwd=GTEST_ROOT。
if ! ( cd "$GTEST_ROOT" && g++ -std=c++17 -I"$GTEST_INC" -I"$GTEST_ROOT" -w \
        -c "$GTEST_SRC/gtest-all.cc" -o "$BUILD_DIR/gtest-all.o" ) 2>"$BUILD_DIR/gtest.err"; then
    err "gtest 编译失败"; cat "$BUILD_DIR/gtest.err"; exit 1
fi
if ! ( cd "$GTEST_ROOT" && g++ -std=c++17 -I"$GTEST_INC" -I"$GTEST_ROOT" -w \
        -c "$GTEST_SRC/gtest_main.cc" -o "$BUILD_DIR/gtest_main.o" ) 2>>"$BUILD_DIR/gtest.err"; then
    err "gtest_main 编译失败"; cat "$BUILD_DIR/gtest.err"; exit 1
fi
ar rcs "$BUILD_DIR/libgtest.a" "$BUILD_DIR/gtest-all.o" "$BUILD_DIR/gtest_main.o"
ok "libgtest.a 就绪"

# ============================================================================
# 4. 编译生产源码（一次，带覆盖率插桩）
# ============================================================================
info "编译生产源码..."
PROD_OBJS=()
fail=0
for src in "${PROD_SOURCES[@]}"; do
    base="$(basename "${src%.cpp}")"
    obj="$BUILD_DIR/objs/${base}.o"
    if needs_rebuild "$src" "$obj"; then
        if g++ "${CXX_FLAGS[@]}" -c "$src" -o "$obj" 2>"$BUILD_DIR/${base}.err"; then
            PROD_OBJS+=("$obj")
        else
            err "编译失败: $src"; tail -5 "$BUILD_DIR/${base}.err"
            fail=1
        fi
    else
        ok "$base 已缓存，跳过"
        PROD_OBJS+=("$obj")
    fi
done
[[ $fail -ne 0 ]] && { err "生产源码编译失败，终止"; exit 1; }
ok "生产源码编译完成: ${#PROD_OBJS[@]} 个目标文件"

# ============================================================================
# 5. 逐个编译 + 运行测试
# ============================================================================
info "编译并运行测试用例..."

TOTAL_TESTS=0; TOTAL_PASSED=0; TOTAL_FAILED=0
COMPILE_FAIL=0; RUN_FAIL=0
RESULTS_FILE="$BUILD_DIR/results.txt"
: > "$RESULTS_FILE"

# 在 build 目录中运行，使 .gcda 落在 .o 旁边（gcov 能找到）
cd "$BUILD_DIR/objs" || { err "无法进入 $BUILD_DIR/objs"; exit 1; }

for tsrc in "${TEST_SOURCES[@]}"; do
    tname="$(basename "${tsrc%.cpp}")"
    tobj="$BUILD_DIR/objs/${tname}.o"
    tbin="$BUILD_DIR/bin/${tname}"

    # 编译（增量：源比 .o 新才重编）
    if needs_rebuild "$tsrc" "$tobj"; then
        if ! g++ "${CXX_FLAGS[@]}" -c "$tsrc" -o "$tobj" 2>"$BUILD_DIR/${tname}.err"; then
            err "编译失败: $tname"; tail -8 "$BUILD_DIR/${tname}.err"
            COMPILE_FAIL=$((COMPILE_FAIL + 1))
            echo "COMPILE_FAIL  $tname" >> "$RESULTS_FILE"
            continue
        fi
    fi

    # 链接（增量：.o 比 bin 新才重链）
    if needs_rebuild "$tobj" "$tbin"; then
        if ! g++ "${LINK_FLAGS[@]}" "$tobj" "${PROD_OBJS[@]}" "$BUILD_DIR/libgtest.a" \
                $GCOV_LINK_FLAG -o "$tbin" 2>>"$BUILD_DIR/${tname}.err"; then
            err "链接失败: $tname"; tail -8 "$BUILD_DIR/${tname}.err"
            COMPILE_FAIL=$((COMPILE_FAIL + 1))
            echo "LINK_FAIL  $tname" >> "$RESULTS_FILE"
            continue
        fi
    fi

    # 运行 —— 用子 shell + 重定向捕获输出，避免崩溃测试的信号影响脚本本身。
    # timeout 兜底防止死锁/卡死（180s 足够覆盖最慢用例）。
    export LD_LIBRARY_PATH="$TORCH_LIB:$PYTHON_LIB_DIR:${LD_LIBRARY_PATH:-}"
    run_out="$BUILD_DIR/${tname}.run"
    : > "$run_out"
    if [[ $VERBOSE -eq 1 ]]; then
        timeout --preserve-status 180s "$tbin" --gtest_color=yes >"$run_out" 2>&1
    else
        timeout --preserve-status 180s "$tbin" --gtest_brief=1 --gtest_color=no >"$run_out" 2>&1
    fi
    rc=$?

    # 从输出解析用例数与通过数。gtest 输出形如：
    #   [==========] 8 tests from 1 test suite ran. (47 ms total)
    #   [  PASSED  ] 8 tests.
    #   [  FAILED  ] 2 tests
    n_tests=$(grep -oP '\[==========\] \K[0-9]+(?= tests)' "$run_out" 2>/dev/null | head -1)
    n_passed=$(grep -oP '\[  PASSED  \] \K[0-9]+(?= tests)' "$run_out" 2>/dev/null | head -1)
    n_failed=$(grep -oP '\[  FAILED  \] \K[0-9]+(?= tests)' "$run_out" 2>/dev/null | head -1)
    [[ -z "$n_tests" ]] && n_tests=0
    [[ -z "$n_passed" ]] && n_passed=0
    [[ -z "$n_failed" ]] && n_failed=0
    # 如果 failed 未打印但 tests > passed，差额算 failed
    if [[ $n_failed -eq 0 && $n_tests -gt $n_passed ]]; then
        n_failed=$((n_tests - n_passed))
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + n_tests))
    TOTAL_PASSED=$((TOTAL_PASSED + n_passed))
    TOTAL_FAILED=$((TOTAL_FAILED + n_failed))

    if [[ $rc -eq 0 ]]; then
        ok "$tname: $n_passed/$n_tests 通过"
        echo "PASS  $tname  $n_passed/$n_tests" >> "$RESULTS_FILE"
    else
        err "$tname: $n_failed 失败 (共 $n_tests)"
        [[ $VERBOSE -eq 0 ]] && tail -5 "$run_out"
        RUN_FAIL=$((RUN_FAIL + 1))
        echo "FAIL  $tname  $n_passed/$n_tests" >> "$RESULTS_FILE"
    fi
done

# ============================================================================
# 6. 覆盖率统计（gcov）
# ============================================================================
COV_LINE_PCT=0; COV_BRANCH_PCT=0
COV_LINES_TOTAL=0; COV_LINES_HIT=0
COV_BRANCH_TOTAL=0; COV_BRANCH_HIT=0

if [[ $DO_COV -eq 1 ]]; then
    info "统计覆盖率..."
    cd "$BUILD_DIR/objs" || true

    COV_LOG="$BUILD_DIR/coverage_gcov.txt"
    : > "$COV_LOG"

    # 对生产源码运行 gcov。kvcache_hbm_manger.h 是头文件类，其 HBMManager 方法
    # 的覆盖率数据由调用它的测试 TU（test_hbm_manager.cpp，本脚本也以 --coverage
    # 插桩编译）产生；因此额外对 test_hbm_manager.cpp 运行 gcov，使其 hbm 头文件
    # 覆盖率被统计到（gcov 会报告该 TU 包含的所有文件，含 kvcache_hbm_manger.h）。
    for src in "${PROD_SOURCES[@]}"; do
        gcov -b -o "$BUILD_DIR/objs" "${src}" >> "$COV_LOG" 2>&1 || true
    done
    # 额外对驱动 HBMManager 覆盖率的测试 TU 运行 gcov（其 hbm 头文件覆盖率高）。
    gcov -b -o "$BUILD_DIR/objs" "$SCRIPT_DIR/ut/test_hbm_manager.cpp" >> "$COV_LOG" 2>&1 || true

    # 解析 gcov 输出，只累计生产源码 .cpp + kvcache_hbm_manger.h（被 inst.cpp
    # 带覆盖率插桩编译，其 HBMManager 内联方法被强制外发为带弧的副本）:
    #   File '/path/to/kvcache_pre.cpp'
    #   Lines executed:95.50% of 1234
    #   Branches executed:87.20% of 567
    # 用状态机：仅在 current_file 为生产 .cpp 或目标头文件时累计其紧随的 Lines/Branches。
    # 每个文件只累计一次（去重）：gcov 会从多个 TU 报告同一头文件，取覆盖率最高者。
    declare -A prod_basenames
    declare -A per_file_lines
    declare -A per_file_branches
    declare -A seen_file
    declare -A seen_branch
    declare -A file_lines_total
    declare -A file_lines_hit
    declare -A file_branch_total
    declare -A file_branch_hit
    for src in "${PROD_SOURCES[@]}"; do prod_basenames["$(basename "${src}")"]=1; done
    # 额外把需要计入覆盖率的头文件加入统计（HBMManager 由 test_hbm_manager.cpp
    # 驱动，其方法被测试 TU 的插桩副本覆盖）。
    prod_basenames["kvcache_hbm_manger.h"]=1
    current_want=0
    while IFS= read -r line; do
        case "$line" in
            File\ *\'*\')
                # 任何 File '...' 行：先重置 want，再判断该文件是否为生产 .cpp
                current_want=0
                fpath="${line#File \'}"; fpath="${fpath%%\'*}"
                base="${fpath##*/}"
                if [[ -n "${prod_basenames[$base]:-}" ]]; then
                    current_want=1
                    LAST_FILE="$base"
                    # 记录该文件在本 gcov 运行中的行/分支（每个文件首条 Lines/Branches 生效）
                fi
                ;;
            Lines\ executed:*)
                if [[ $current_want -eq 1 ]]; then
                    pct=$(echo "$line" | grep -oP 'Lines executed:\K[0-9.]+')
                    total=$(echo "$line" | grep -oP 'of \K[0-9]+')
                    if [[ -n "$total" && "$total" -gt 0 ]]; then
                        # 记录该文件的行覆盖率。对于头文件（如 kvcache_hbm_manger.h），
                        # 不同 TU 可能报告不同的行总数；取覆盖行数最多（即最完整）的版本。
                        new_hit=$(awk -v t="$total" -v p="$pct" 'BEGIN{ printf "%.4f", t*p/100 }')
                        if [[ -z "${seen_file[$LAST_FILE]:-}" ]]; then
                            COV_LINES_TOTAL=$((COV_LINES_TOTAL + total))
                            COV_LINES_HIT=$(awk -v h="$COV_LINES_HIT" -v t="$total" -v p="$pct" \
                                'BEGIN{ printf "%.4f", h + t*p/100 }')
                            seen_file["$LAST_FILE"]="$total:$new_hit"
                        else
                            old_total="${seen_file[$LAST_FILE]%%:*}"
                            old_hit="${seen_file[$LAST_FILE]#*:}"
                            if awk -v n="$new_hit" -v o="$old_hit" 'BEGIN{ exit !(n > o) }'; then
                                COV_LINES_TOTAL=$((COV_LINES_TOTAL + total - old_total))
                                COV_LINES_HIT=$(awk -v h="$COV_LINES_HIT" -v n="$new_hit" -v o="$old_hit" \
                                    'BEGIN{ printf "%.4f", h + n - o }')
                                seen_file["$LAST_FILE"]="$total:$new_hit"
                            fi
                        fi
                        per_file_lines["$LAST_FILE"]="${pct}% (${total} 行)"
                    fi
                fi
                ;;
            Branches\ executed:*)
                if [[ $current_want -eq 1 ]]; then
                    pct=$(echo "$line" | grep -oP 'Branches executed:\K[0-9.]+')
                    total=$(echo "$line" | grep -oP 'of \K[0-9]+')
                    if [[ -n "$total" && "$total" -gt 0 ]]; then
                        new_hit=$(awk -v t="$total" -v p="$pct" 'BEGIN{ printf "%.4f", t*p/100 }')
                        if [[ -z "${seen_branch[$LAST_FILE]:-}" ]]; then
                            COV_BRANCH_TOTAL=$((COV_BRANCH_TOTAL + total))
                            COV_BRANCH_HIT=$(awk -v h="$COV_BRANCH_HIT" -v t="$total" -v p="$pct" \
                                'BEGIN{ printf "%.4f", h + t*p/100 }')
                            seen_branch["$LAST_FILE"]="$total:$new_hit"
                        else
                            old_total="${seen_branch[$LAST_FILE]%%:*}"
                            old_hit="${seen_branch[$LAST_FILE]#*:}"
                            if awk -v n="$new_hit" -v o="$old_hit" 'BEGIN{ exit !(n > o) }'; then
                                COV_BRANCH_TOTAL=$((COV_BRANCH_TOTAL + total - old_total))
                                COV_BRANCH_HIT=$(awk -v h="$COV_BRANCH_HIT" -v n="$new_hit" -v o="$old_hit" \
                                    'BEGIN{ printf "%.4f", h + n - o }')
                                seen_branch["$LAST_FILE"]="$total:$new_hit"
                            fi
                        fi
                        per_file_branches["$LAST_FILE"]="${pct}% (${total} 分支)"
                    fi
                fi
                ;;
        esac
    done < "$COV_LOG"
    unset awk_parse

    if [[ $COV_LINES_TOTAL -gt 0 ]]; then
        COV_LINE_PCT=$(echo "scale=2; $COV_LINES_HIT * 100 / $COV_LINES_TOTAL" | bc -l 2>/dev/null || echo "0")
    fi
    if [[ $COV_BRANCH_TOTAL -gt 0 ]]; then
        COV_BRANCH_PCT=$(echo "scale=2; $COV_BRANCH_HIT * 100 / $COV_BRANCH_TOTAL" | bc -l 2>/dev/null || echo "0")
    fi
    ok "覆盖率统计完成"
fi

# ============================================================================
# 7. 汇总报告
# ============================================================================
echo ""
echo "══════════════════════════════════════════════════════════════════════════════"
echo "                    gsa_prefetch 测试报告"
echo "══════════════════════════════════════════════════════════════════════════════"
echo ""
printf "  ${BOLD}测试用例统计${NC}\n"
printf "  ─────────────────────────────────────────────\n"
printf "  测试文件总数   : %d\n" "${#TEST_SOURCES[@]}"
printf "  编译失败       : %d\n" "$COMPILE_FAIL"
printf "  运行失败       : %d\n" "$RUN_FAIL"
printf "  用例总数 (TEST): %d\n" "$TOTAL_TESTS"
printf "  ${GREEN}通过数          : %d${NC}\n" "$TOTAL_PASSED"
if [[ $TOTAL_FAILED -gt 0 ]]; then
    printf "  ${RED}失败数          : %d${NC}\n" "$TOTAL_FAILED"
else
    printf "  失败数          : %d\n" "$TOTAL_FAILED"
fi
printf "  通过率          : %.1f%%\n" \
    "$(echo "scale=1; $TOTAL_PASSED * 100 / ($TOTAL_TESTS + 0.001)" | bc -l 2>/dev/null || echo 0)"
echo ""

if [[ $DO_COV -eq 1 ]]; then
    printf "  ${BOLD}覆盖率统计 (生产源码 .cpp)${NC}\n"
    printf "  ─────────────────────────────────────────────\n"
    printf "  行覆盖率        : ${GREEN}%.2f%%${NC}  (%d / %d 行)\n" \
        "$COV_LINE_PCT" "${COV_LINES_HIT%.*}" "$COV_LINES_TOTAL" 2>/dev/null || \
        printf "  行覆盖率        : %.2f%%\n" "$COV_LINE_PCT"
    printf "  分支覆盖率      : ${GREEN}%.2f%%${NC}  (%d / %d 分支)\n" \
        "$COV_BRANCH_PCT" "${COV_BRANCH_HIT%.*}" "$COV_BRANCH_TOTAL" 2>/dev/null || \
        printf "  分支覆盖率      : %.2f%%\n" "$COV_BRANCH_PCT"
    echo ""
    printf "  逐文件明细 (生产源码 .cpp + kvcache_hbm_manger.h):\n"
    for src in "${PROD_SOURCES[@]}"; do
        base="$(basename "${src}")"
        lp="${per_file_lines[$base]:-N/A}"
        bp="${per_file_branches[$base]:-N/A}"
        printf "    %-30s 行=%-22s 分支=%-22s\n" "$base" "$lp" "$bp"
    done
    # 额外展示插桩头文件的覆盖率（由 kvcache_hbm_manger_inst.cpp 带插桩包含）。
    lp="${per_file_lines[kvcache_hbm_manger.h]:-N/A}"
    bp="${per_file_branches[kvcache_hbm_manger.h]:-N/A}"
    printf "    %-30s 行=%-22s 分支=%-22s\n" "kvcache_hbm_manger.h" "$lp" "$bp"
fi

echo ""
echo "══════════════════════════════════════════════════════════════════════════════"
if [[ $COMPILE_FAIL -eq 0 && $RUN_FAIL -eq 0 && $TOTAL_FAILED -eq 0 ]]; then
    ok "${BOLD}全部通过！${NC}"
else
    warn "存在失败项，请检查 $BUILD_DIR/ 下的 .err / .run 日志"
fi
echo ""

# 逐条明细（可选 verbose）
if [[ $VERBOSE -eq 1 ]]; then
    echo "逐文件结果:"
    cat "$RESULTS_FILE"
fi

exit $((COMPILE_FAIL + RUN_FAIL + TOTAL_FAILED > 0 ? 1 : 0))
