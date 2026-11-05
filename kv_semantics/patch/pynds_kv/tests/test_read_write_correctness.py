import argparse
import mmap
import sys

import numpy as np
import pynds_kv

parser = argparse.ArgumentParser(description="KV读写一致性验证测试")
parser.add_argument("--num_blocks", type=int, default=128, help="每批次block数量")
parser.add_argument(
    "--token_size", type=int, default=128 * 1024, help="单个token大小(字节), 默认128KB"
)
parser.add_argument(
    "--num_tokens_per_block",
    type=int,
    default=256,
    help="每个block中token数量, 默认256",
)
parser.add_argument(
    "--nds_config_path", type=str, default=None, help="NDS 配置文件路径"
)
parser.add_argument(
    "--nsid",
    type=int,
    default=3618096867,
    help="NDS namespace id；盘框不覆盖写，每轮测试应换不同 nsid 避免跨轮 slot 污染",
)
parser.add_argument(
    "--key_high",
    type=int,
    default=0,
    help="80-bit block_key 的高 16 位（0-65535），模拟 tp_rank 编码场景；"
    "所有 block_id 会加上 key_high << 64，用于验证 80-bit key 读写查一致性",
)
args = parser.parse_args()

if not 0 <= args.key_high <= 0xFFFF:
    print(f"!!! 错误: --key_high 必须在 [0, 65535]，当前为 {args.key_high}")
    sys.exit(1)

NUM_BLOCKS = args.num_blocks
TOKEN_SIZE = args.token_size
NUM_TOKENS_PER_BLOCK = args.num_tokens_per_block
ONE_LAYER_CACHE_SIZE = TOKEN_SIZE * NUM_TOKENS_PER_BLOCK
TOTAL_SIZE = NUM_BLOCKS * ONE_LAYER_CACHE_SIZE
NSID_G = args.nsid
PATTERNS = ["sequential", "alternating", "incremental"]


def generate_expected(block_idx, size, pattern_name):
    if pattern_name == "sequential":
        return (np.arange(size, dtype=np.uint32) % 251 + 1).astype(np.uint8)
    elif pattern_name == "alternating":
        return np.where(np.arange(size) % 2 == 0, 1, 0).astype(np.uint8)
    elif pattern_name == "incremental":
        return ((np.arange(size, dtype=np.uint32) + block_idx) % 256).astype(np.uint8)
    raise ValueError(f"未知模式: {pattern_name}")


def fill_buffer_with_pattern(buf, blocks, pattern_name):
    for i, block in enumerate(blocks):
        expected = generate_expected(i, ONE_LAYER_CACHE_SIZE, pattern_name)
        block[:] = expected


def load_block_ids_from_file():
    block_ids = []
    print("    正在从key.txt加载blockID...")
    try:
        with open("key.txt", "r") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError("key.txt文件为空")
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        block_ids.append(int(line))
                    except ValueError:
                        print(f"    警告: 跳过无效行：{line}")
            if not block_ids:
                raise ValueError("未找到有效的blockID")
            print(f"    成功加载 {len(block_ids)} 个blockID")
    except FileNotFoundError:
        print("    未找到key.txt文件，使用顺序blockID 0..N-1")
        block_ids = list(range(NUM_BLOCKS))
    except Exception as e:
        print(f"    加载blockids失败：{e}，使用顺序blockID 0..N-1")
        block_ids = list(range(NUM_BLOCKS))
    return block_ids[:NUM_BLOCKS]


def run_correctness_test():
    print("=" * 80)
    print("KV读写一致性验证测试".center(80))
    print("=" * 80)
    print(f"NUM_BLOCKS:           {NUM_BLOCKS}")
    print(f"TOKEN_SIZE:           {TOKEN_SIZE} 字节 ({TOKEN_SIZE / 1024:.0f} KB)")
    print(f"NUM_TOKENS_PER_BLOCK: {NUM_TOKENS_PER_BLOCK}")
    print(
        f"ONE_LAYER_CACHE_SIZE: {ONE_LAYER_CACHE_SIZE} 字节 ({ONE_LAYER_CACHE_SIZE / 1024 / 1024:.2f} MB)"
    )
    print(
        f"TOTAL_SIZE:           {TOTAL_SIZE} 字节 ({TOTAL_SIZE / 1024 / 1024:.2f} MB)"
    )
    print(f"NSID:                 {NSID_G}")
    print(f"测试模式:             {PATTERNS}")
    print("=" * 80)

    block_ids = load_block_ids_from_file()
    if args.key_high != 0:
        # 80-bit key：高 16 位编入 key_high（模拟 tp_rank 编码），低 64 位保持原样
        block_ids = [bid | (args.key_high << 64) for bid in block_ids]
        print(
            f"    已启用 80-bit key: key_high={args.key_high}，示例 block_id={block_ids[0]}"
        )
    if len(block_ids) < NUM_BLOCKS:
        print(
            f"    警告: blockID数量({len(block_ids)})不足({NUM_BLOCKS}), 减少测试block数"
        )
        NUM_BLOCKS_ACTUAL = len(block_ids)
    else:
        NUM_BLOCKS_ACTUAL = NUM_BLOCKS

    offsets = [0] * NUM_BLOCKS_ACTUAL
    lengths = [ONE_LAYER_CACHE_SIZE] * NUM_BLOCKS_ACTUAL
    nsids = [NSID_G] * NUM_BLOCKS_ACTUAL

    print(">>> Phase 0: 80-bit key 边界校验（非法 key 应抛异常，无需 init）")
    for bad_key, desc in [(1 << 80, "2^80 超界"), (-1, "负数")]:
        try:
            pynds_kv.is_exists([bad_key], NSID_G)
            print(f"!!! 错误: {desc} key 未被拒绝")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            print(f"    {desc} key 被正确拒绝: {e}")

    print(f">>> Phase I: 分配 {TOTAL_SIZE} 字节的页面对齐缓冲区")
    mm = mmap.mmap(-1, TOTAL_SIZE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    buf = np.frombuffer(mm, dtype=np.uint8, count=TOTAL_SIZE)
    addr = buf.__array_interface__["data"][0]
    print(f"    缓冲区地址 = 0x{addr:x}")
    print(f"    4096字节对齐? {'YES' if addr % 4096 == 0 else 'NO'}")

    print(f">>> Phase II: 调用pynds_kv.init()")
    if args.nds_config_path is None:
        print("!!! 错误: 必须提供 --nds_config_path")
        mm.close()
        return

    nds_config_path = args.nds_config_path

    ret = pynds_kv.init(buf, TOTAL_SIZE, nds_config_path)
    print(f"    init() 返回值: {ret}")
    if ret != 0:
        print("!!! 初始化失败")
        mm.close()
        return

    try:
        blocks = []
        for i in range(NUM_BLOCKS_ACTUAL):
            offset = i * ONE_LAYER_CACHE_SIZE
            block = buf[offset : offset + ONE_LAYER_CACHE_SIZE]
            blocks.append(block)
        print(f"    block预生成完成, 共{NUM_BLOCKS_ACTUAL}个block")

        results = {}

        for pattern_name in PATTERNS:
            print("=" * 80)
            print(f">>> 测试模式: {pattern_name}")
            print(f"    步骤1: 填充缓冲区为 {pattern_name} 模式")
            fill_buffer_with_pattern(buf, blocks, pattern_name)

            print(f"    步骤2: batchPut 写入 {NUM_BLOCKS_ACTUAL} 个block")
            ret = pynds_kv.batchPut(block_ids, blocks, offsets, lengths, nsids)
            if ret != 0:
                print(f"!!! batchPut失败, 返回值: {ret}")
                results[pattern_name] = False
                continue
            print(f"    batchPut成功, 返回值: {ret}")

            print(f"    步骤2a: is_exists 检查 block 存在性 (nsid={NSID_G})")
            exists_count = pynds_kv.is_exists(block_ids, NSID_G)
            if exists_count != NUM_BLOCKS_ACTUAL:
                print(
                    f"!!! is_exists 返回 {exists_count}/{NUM_BLOCKS_ACTUAL}, 存在性检查失败"
                )
                results[pattern_name] = False
                continue
            print(f"    is_exists 通过, 全部 {NUM_BLOCKS_ACTUAL} 个 block 存在")

            print(f"    步骤2b: is_exists 用错误 nsid 验证隔离性 (nsid=999)")
            wrong_exists = pynds_kv.is_exists(block_ids, 999)
            if wrong_exists != 0:
                print(
                    f"!!! is_exists 用错误 nsid 返回 {wrong_exists}, 预期 0, 隔离性检查失败"
                )
                results[pattern_name] = False
                continue
            print(f"    隔离性通过, 错误 nsid 返回 0")

            print(f"    步骤3: 清零缓冲区")
            buf[:] = 0

            print(f"    步骤4: batchGet 读取 {NUM_BLOCKS_ACTUAL} 个block")
            ret = pynds_kv.batchGet(block_ids, blocks, offsets, lengths, nsids)
            if ret != 0:
                print(f"!!! batchGet失败, 返回值: {ret}")
                results[pattern_name] = False
                continue
            print(f"    batchGet成功, 返回值: {ret}")

            print(f"    步骤5: 验证读取数据是否符合 {pattern_name} 模式")
            total_mismatch_bytes = 0
            failed_blocks = 0
            for i in range(NUM_BLOCKS_ACTUAL):
                expected = generate_expected(i, ONE_LAYER_CACHE_SIZE, pattern_name)
                actual = blocks[i]
                if not np.array_equal(actual, expected):
                    mismatch = np.sum(actual != expected)
                    total_mismatch_bytes += mismatch
                    failed_blocks += 1
                    first_mismatch_idx = np.where(actual != expected)[0][0]
                    print(
                        f"    block[{i}] 不匹配: 预期={expected[first_mismatch_idx]}, 实际={actual[first_mismatch_idx]}, "
                        f"位置={first_mismatch_idx}, 不匹配字节数={mismatch}"
                    )

            if failed_blocks == 0:
                print(
                    f"    ✓ {pattern_name} 模式验证通过! 所有 {NUM_BLOCKS_ACTUAL} 个block数据一致"
                )
                results[pattern_name] = True
            else:
                print(
                    f"    ✗ {pattern_name} 模式验证失败! {failed_blocks}/{NUM_BLOCKS_ACTUAL} 个block不匹配, "
                    f"总不匹配字节数={total_mismatch_bytes}"
                )
                results[pattern_name] = False

        print("=" * 80)
        print("最终汇总".center(80))
        print("=" * 80)
        all_passed = True
        for pattern_name in PATTERNS:
            status = "通过" if results.get(pattern_name, False) else "失败"
            symbol = "✓" if results.get(pattern_name, False) else "✗"
            print(f"    {symbol} {pattern_name}: {status}")
            if not results.get(pattern_name, False):
                all_passed = False
        print("=" * 80)
        if all_passed:
            print("所有模式验证通过!".center(80))
        else:
            print("存在验证失败的模式!".center(80))
        print("=" * 80)
    finally:
        mm.close()


if __name__ == "__main__":
    try:
        run_correctness_test()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常：{e}")
