# -*- coding: utf-8 -*-
"""C-splitKV on-device 基准测试（需在 A2/A3 NPU 机器上运行）。

前置条件:
  1. 已编译安装 kernel + whl:  cd ops && bash build.sh a3 install
  2. import gsa_ops 可用, torch.npu.is_available() == True

测试内容:
  * 精度: splitKV 开(自动)/关(QSFA_DISABLE_SPLITKV=1)与 fp32 参考实现对比;
          开/关两态互相对比(combine 数学正确性的主要判据)。
          (注意: mm1Res/mm2Res 中间量为 fp16——D 优化, 对比容差已按 dtype 放宽)
  * 性能: bs 扫描, 输出 P/尾波任务数 决策(与 host DecideSplitKV 同模型)、
          off/auto 两态耗时与加速比。

生产签名对齐:
  * layout_query=TND, layout_kv=PA_BSND, sparse_mode=3, attention_mode=2(MLA-absorb),
    quant_scale_repo_mode=1(COMBINE), tile_size=128, rope_head_dim=64。
  * TND query [T=bs, gsize, 576] (decode 每 batch 1 token); actual_seq_lens_q 为前缀和。
  * PA_BSND packed KV [totalBlocks, 128, 1, 656] + block_table [bs, blocksPerBatch]。

实现说明:
  * 开/关两态分别跑在独立子进程中(env 在 tiling 前设置)，
    规避 aclnn 单算子路径下任何 tiling/编译缓存对 A/B 的干扰。
  * key=value 同一 packed int8 张量（MLA c_kv, COMBINE 模式）:
      字节 [0,512)   nope int8 (512)
      字节 [512,640) rope 64 x fp16/bf16 (128)
      字节 [640,656) 4 x fp32 per-tile(128) 反量化 scale
  * 有效 sparse 布局为“稠密前缀 + 尾部 -1 padding”（P1 effectiveSparseCount 语义）。
  * splitKV 启用门控(与 host DecideSplitKV 一致): 非A5、qLayout∈{BSND,TND}、
    gSize<=256、s2Loops>=4、建模收益>=25%、subTotal<=4*aicNum、P<=8。

用法:
  python ut/bench_splitkv_gsa.py                       # 默认扫 bs(含 64/112/128 大batch)
  python ut/bench_splitkv_gsa.py --bs 1,2,4,8          # 指定 bs 列表
  python ut/bench_splitkv_gsa.py --check-only          # 只做精度
  python ut/bench_splitkv_gsa.py --force-p 4           # 强制全任务切P=4 验证 combine 路径
  python ut/bench_splitkv_gsa.py --random-scale        # 随机 per-tile 反量化 scale
"""
import argparse
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

import torch

HEAD_DIM = 512
HEAD_DIM_ROPE = 64
PACK_DIM = 656  # 512(int8 nope) + 64*2(rope, fp16/bf16) + 4*4(fp32 scale)
ROPE_BYTES = 128  # 64 x 2B(fp16/bf16 rope 区字节跨度)
S_INNER = 512  # kernel s2BaseSize
BLOCK_SIZE = 128  # PA block size(=生产 tile_size)
AIC_NUM = 24  # A2/A3 cube 核数(GetBlockNum())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="bench splitKV for kv_quant_sparse_flash_attention_gsa")
    p.add_argument("--bs", type=str, default="1,2,4,8,12,16,24,25,30,36,48,64,112,128")
    p.add_argument("--kv-len", type=int, default=32768)
    p.add_argument("--sparse-count", type=int, default=5120, help="sparse_indices 最后一维 K")
    p.add_argument("--valid-count", type=int, default=0,
                   help="有效(非-1)索引个数, 0=全部有效; 尾部自动填 -1 padding")
    p.add_argument("--gsize", type=int, default=32, help="n1/n2 (q heads per kv group)")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--scale-value", type=float, default=0.0, help="0 表示 1/sqrt(576)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--force-p", type=int, default=0, help=">0 时设置 QSFA_FORCE_SPLITKV 强制 P")
    p.add_argument("--random-scale", action="store_true", help="随机 per-tile scale(默认全 1.0)")
    # 内部参数(子进程模式)
    p.add_argument("--state", type=str, default=None, choices=[None, "off", "auto"],
                   help=argparse.SUPPRESS)
    p.add_argument("--dump", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--timed", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.scale_value == 0.0:
        args.scale_value = 1.0 / math.sqrt(HEAD_DIM + HEAD_DIM_ROPE)
    return args


def expected_splitkv_p(task_num, s2_loops, aic=AIC_NUM):
    """复刻 host DecideSplitKV(C: 尾波自适应+WCP加权分核)的搜索模型。
    返回 (P, tailTasks, modeledMakespan)；P=1 表示不启用。
    WCP公式与 host/kernel 严格一致: sub归属核 = (prefix-1)*aic/totalW。"""
    if s2_loops < 4 or task_num == 0:
        return 1, 0, 0
    base = ((task_num + aic - 1) // aic) * s2_loops
    sub_budget = 4 * aic
    max_p = 8

    def simulate(x, p):
        part_w = (s2_loops + p - 1) // p
        total_w = (task_num - x) * s2_loops + x * part_w * p
        if total_w == 0:
            return 0
        acc = 0
        core = -1
        core_load = 0
        mx = 0
        for t in range(task_num):
            pn = p if t >= task_num - x else 1
            w = s2_loops if pn == 1 else part_w
            for _ in range(pn):
                acc += w
                c = (acc - 1) * aic // total_w
                if c != core:
                    core = c
                    core_load = 0
                core_load += w
                mx = max(mx, core_load)
        return mx

    tail_base = task_num % aic
    if tail_base == 0:
        tail_base = min(aic, task_num)
    candidates = list(range(tail_base, task_num + 1, aic))
    if not candidates or candidates[-1] != task_num:
        candidates.append(task_num)
    best_p, best_tail, best_mk, best_sub = 1, 0, base, 0
    for x in candidates:
        if x == 0 or x > task_num:
            continue
        sub_for_p = (sub_budget - (task_num - x)) // x if sub_budget > (task_num - x) else 0
        p_max = min(s2_loops, max_p, sub_for_p)
        for p in range(2, p_max + 1):
            sub_total = (task_num - x) + x * p
            if sub_total > sub_budget:
                continue
            mk = simulate(x, p)
            if mk < best_mk or (mk == best_mk and best_sub > 0 and sub_total < best_sub):
                best_p, best_tail, best_mk, best_sub = p, x, mk, sub_total
    if best_p >= 2 and best_mk * 4 <= base * 3:
        return best_p, best_tail, best_mk
    return 1, 0, base


def make_inputs(args, bs, device):
    """生产签名输入: TND query(decode, T=bs) + PA_BSND packed KV + block_table。
    TND语义: actual_seq_lens_query是前缀和(kernel按差分取每batch长度); sparse_indices=[T,1,K]。"""
    torch.manual_seed(args.seed + bs)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # TND query: [T=bs*1, gsize, 576] (decode: 每batch 1个token)
    query = (torch.randn(bs, args.gsize, HEAD_DIM + HEAD_DIM_ROPE, dtype=torch.float32) * 0.05).to(dt)

    # PA_BSND packed KV: [totalBlocks, blockSize, 1(=kvHeadNum), PACK_DIM]
    blocks_per_batch = (args.kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = bs * blocks_per_batch
    kv = torch.randint(-64, 63, (total_blocks, BLOCK_SIZE, 1, PACK_DIM), dtype=torch.int8)
    kv_bytes = kv.view(torch.uint8)
    rope_begin = HEAD_DIM
    rope_end = rope_begin + ROPE_BYTES
    # rope: 字节 [512, 640) 按 dt 解释
    rope = (torch.randn(total_blocks, BLOCK_SIZE, 1, HEAD_DIM_ROPE, dtype=torch.float32) * 0.05).to(dt)
    kv_bytes[..., rope_begin:rope_end] = rope.view(torch.uint8).reshape(total_blocks, BLOCK_SIZE, 1, -1)
    # per-tile scale: 字节 [640, 656) 4 x fp32
    if args.random_scale:
        scales = torch.rand(total_blocks, BLOCK_SIZE, 1, 4, dtype=torch.float32) * 0.5 + 0.5
    else:
        scales = torch.ones(total_blocks, BLOCK_SIZE, 1, 4, dtype=torch.float32)
    kv_bytes[..., rope_end:] = scales.view(torch.uint8).reshape(total_blocks, BLOCK_SIZE, 1, -1)

    # block_table: [bs, blocksPerBatch] int32 (连续块编号, token i -> bt[b, i//128]块的 i%128行)
    block_table = torch.arange(total_blocks, dtype=torch.int32).reshape(bs, blocks_per_batch)

    # sparse_indices: TND [T=bs, 1, K], 稠密前缀 + 尾部 -1
    valid = args.valid_count if args.valid_count > 0 else args.sparse_count
    valid = min(valid, args.sparse_count)
    indices = torch.full((bs, 1, args.sparse_count), -1, dtype=torch.int32)
    for b in range(bs):
        sel = torch.randperm(args.kv_len)[:valid].sort().values.to(torch.int32)
        indices[b, 0, :valid] = sel

    # TND: actual_seq_lens_q 为前缀和(decode 每 batch 1 token -> [1,2,...,bs])
    seq_q = torch.cumsum(torch.ones(bs, dtype=torch.int32), 0).to(torch.int32)
    seq_kv = torch.full((bs,), args.kv_len, dtype=torch.int32)
    return query.to(device), kv.to(device), indices.to(device), seq_q.to(device), \
        seq_kv.to(device), block_table.to(device)


def run_op(args, query, kv, indices, seq_q, seq_kv, block_table):
    # 生产签名: layout_query=TND, layout_kv=PA_BSND, sparse_mode=3, attention_mode=2(MLA-absorb),
    # quant_scale_repo_mode=1(COMBINE packed), pre/next tokens INT64_MAX(mode3下内核自行推导)
    out = torch.ops._C_gsa.npu_kv_quant_sparse_flash_attention_gsa(
        query, kv, kv, indices,
        args.scale_value, 2, 2,            # key/value quant mode: 2=per-tile
        None, None, block_table,           # dequant scale 走 COMBINE packed; PA_BSND 需 block_table
        seq_q, seq_kv,
        None, None,                        # sinks, actualSparseLengths
        1,                                 # sparse_block_size
        "TND", "PA_BSND",
        3,                                 # sparse_mode: right-down causal
        9223372036854775807, 9223372036854775807,  # pre/next tokens (mode3 下内核自行推导)
        2, 1,                              # attention_mode=2(MLA-absorb), quant_scale_repo_mode=COMBINE
        128, 64, False)                    # tile_size, rope_head_dim, return_softmax_lse
    return out[0] if isinstance(out, tuple) else out


def reference(args, query, kv, indices, block_table):
    """fp32 参考实现(TND/PA语义, MLA-absorb: V=dequant(nope), score 含 rope, scale 作用于全点积)。"""
    dt = query.dtype
    bs, g, _ = query.shape  # TND decode: T=bs, 每 batch 1 token
    q = query.float().cpu()
    kv_cpu = kv.cpu()  # int8 [totalBlocks, 128, 1, 656]
    bt = block_table.cpu().long()
    idx_cpu = indices.cpu()  # [bs, 1, K]
    out = torch.empty(bs, g, HEAD_DIM, dtype=torch.float32)
    for b in range(bs):
        idx = idx_cpu[b, 0]
        idx = idx[idx >= 0].long()
        # PA寻址: token i -> kv[bt[b, i//128], i%128]
        blk = bt[b][idx // BLOCK_SIZE]
        sel = kv_cpu[blk, idx % BLOCK_SIZE, 0].contiguous()  # [K, 656] int8
        nope_dq = sel[:, :HEAD_DIM].to(torch.float32)  # int8 -> fp32(带符号)
        scales = sel[:, HEAD_DIM + ROPE_BYTES:] \
            .contiguous().view(torch.float32)  # [K, 4]
        rope = sel[:, HEAD_DIM:HEAD_DIM + ROPE_BYTES] \
            .contiguous().view(dt).to(torch.float32)  # [K, 64]
        nope_dq = nope_dq * scales.repeat_interleave(HEAD_DIM // 4, dim=-1)
        k_n = nope_dq
        k_p = rope
        qq = q[b]  # [G, 576]
        scores = (qq[:, :HEAD_DIM] @ k_n.T + qq[:, HEAD_DIM:] @ k_p.T) * args.scale_value
        p = torch.softmax(scores, dim=-1)
        out[b] = p @ k_n
    return out


def run_child(args, bs):
    """在独立子进程中以指定 state 跑一次算子(保存输出) + 可选计时。"""
    dump = os.path.join(tempfile.gettempdir(), "bench_splitkv_%s_bs%d.pt" % (args.state, bs))
    cmd = [sys.executable, os.path.abspath(__file__),
           "--bs", str(bs), "--kv-len", str(args.kv_len),
           "--sparse-count", str(args.sparse_count), "--valid-count", str(args.valid_count),
           "--gsize", str(args.gsize), "--dtype", args.dtype,
           "--scale-value", str(args.scale_value),
           "--warmup", str(args.warmup), "--iters", str(args.iters),
           "--seed", str(args.seed), "--state", args.state, "--dump", dump]
    if args.force_p > 0:
        cmd += ["--force-p", str(args.force_p)]
    if args.random_scale:
        cmd += ["--random-scale"]
    if not args.check_only and args.state == "auto":
        cmd += ["--timed"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError("child state=%s bs=%d failed" % (args.state, bs))
    ms = None
    for line in proc.stdout.splitlines():
        if line.startswith("MS "):
            ms = float(line.split()[1])
    out = torch.load(dump, map_location="cpu")
    os.unlink(dump)
    return out, ms


def child_main(args):
    """子进程: 设置 env → 构造输入 → 跑一次(存输出) → 可选计时。"""
    if not torch.npu.is_available():
        print("[child] torch.npu 不可用")
        sys.exit(2)
    import gsa_ops  # noqa: F401  加载 torch.ops._C_gsa 注册
    if args.state == "off":
        os.environ["QSFA_DISABLE_SPLITKV"] = "1"
    elif args.force_p > 0:
        os.environ["QSFA_FORCE_SPLITKV"] = str(args.force_p)
    elif args.state == "auto":
        os.environ["QSFA_ENABLE_SPLITKV"] = "1"  # 生产默认关闭自动切分,auto态需显式开启

    device = "npu"
    bs = int(args.bs.split(",")[0])
    query, kv, indices, seq_q, seq_kv, block_table = make_inputs(args, bs, device)
    out = run_op(args, query, kv, indices, seq_q, seq_kv, block_table).float().cpu()
    torch.save(out, args.dump)

    if args.timed:
        for _ in range(args.warmup):
            run_op(args, query, kv, indices, seq_q, seq_kv, block_table)
        torch.npu.synchronize()
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            run_op(args, query, kv, indices, seq_q, seq_kv, block_table)
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        print("MS %.4f" % statistics.median(ts))


def main():
    args = parse_args()
    if args.state is not None:
        child_main(args)
        return

    bs_list = [int(x) for x in args.bs.split(",") if x.strip()]
    valid = args.valid_count if args.valid_count > 0 else args.sparse_count
    s2_loops = (min(valid, args.kv_len) + S_INNER - 1) // S_INNER

    print("[bench] dtype=%s gsize=%d kv_len=%d sparse=%d(valid=%d) scale=%.6f iters=%d" %
          (args.dtype, args.gsize, args.kv_len, args.sparse_count, valid,
           args.scale_value, args.iters))
    if args.force_p > 0:
        print("[bench] QSFA_FORCE_SPLITKV=%d (强制 combine 路径验证)" % args.force_p)
    print("bs\ttaskNum\ts2Loops\tP/tail(model)\toff_ms\tauto_ms\tspeedup")

    for bs in bs_list:
        p_model, tail_model, mk_model = expected_splitkv_p(bs, s2_loops)

        # ---- 两态输出(独立子进程) + fp32 参考 ----
        args.state = "off"
        out_off, _ = run_child(args, bs)
        args.state = "auto"
        out_on, ms_auto = run_child(args, bs)
        args.state = None

        # 参考输入需要在 CPU 上重建(与子进程同种子)
        query, kv, indices, _, _, block_table = make_inputs(args, bs, "cpu")
        ref = reference(args, query, kv, indices, block_table)

        diff_onoff = (out_on - out_off).abs().max().item()
        diff_on_ref = (out_on - ref).abs().max().item()
        diff_off_ref = (out_off - ref).abs().max().item()
        ok_onoff = torch.allclose(out_on, out_off, rtol=1e-2, atol=1e-2)
        tol = 3e-2 if args.dtype == "bf16" else 1.5e-2
        ok_ref = torch.allclose(out_on, ref, rtol=tol, atol=tol)

        status = "PASS" if (ok_onoff and ok_ref) else ("WARN" if ok_onoff else "FAIL")
        print("[check] bs=%-3d on-vs-off max=%.3e(%s) on-vs-ref=%.3e off-vs-ref=%.3e ref_tol=%.0e [%s]" %
              (bs, diff_onoff, "ok" if ok_onoff else "MISMATCH",
               diff_on_ref, diff_off_ref, tol, status))
        if not ok_onoff:
            print("[check] splitKV 开/关结果不一致——combine 路径存在正确性问题, 跳过该 bs 计时")
            continue

        if args.check_only or ms_auto is None:
            continue

        # ---- off 态计时(独立子进程) ----
        args.state = "off"
        _, ms_off = run_child(args, bs)
        args.state = None
        if ms_off is None:
            continue

        speedup = ms_off / ms_auto if ms_auto > 0 else float("inf")
        print("%d\t%d\t%d\t%d/%d\t%.3f\t%.3f\t%.2fx" %
              (bs, bs, s2_loops, p_model, tail_model, ms_off, ms_auto, speedup))

    print("[bench] done. (P/tail(model)=host DecideSplitKV 同模型预期值(P/尾波任务数); "
          "实际决策可用 ASCEND_GLOBAL_LOG_LEVEL=1 从 plog 中 grep 'splitKV enabled')")


if __name__ == "__main__":
    main()
