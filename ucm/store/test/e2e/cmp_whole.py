#!/usr/bin/env python3
import sys
import pathlib
import numpy as np

BLOCK_U16 = 512 * 1024          # 每次 512 K 个 uint16（1 MB 块）

def die(msg):
    print(msg, file=sys.stderr); sys.exit(1)

def cmp_mask(p1: pathlib.Path, p2: pathlib.Path, mask: int = 0xFFFF):
    """
    按掩码比较两个 uint16 二进制文件
    mask: 位为 1 的位才参与比较，默认 0xFFFF 全比较
    返回总差异个数（0 表示完全一致）
    """
    if p1.stat().st_size != p2.stat().st_size:
        die(f"大小不同：{p1.name}={p1.stat().st_size}  {p2.name}={p2.stat().st_size}")
    u16_size = p1.stat().st_size // 2
    total_diff = 0
    off_u16 = 0
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        while off_u16 < u16_size:
            chunk = min(BLOCK_U16, u16_size - off_u16)
            a = np.frombuffer(f1.read(chunk * 2), dtype=np.uint16)
            b = np.frombuffer(f2.read(chunk * 2), dtype=np.uint16)
            # 只比较掩码位
            diff = (a & mask) != (b & mask)
            total_diff += int(np.count_nonzero(diff))
            off_u16 += chunk
    return total_diff

if __name__ == "__main__":
    if len(sys.argv) != 3: die("用法: python cmp_uint16_whole.py <file1> <file2>")
    p1, p2 = map(pathlib.Path, sys.argv[1:3])
    if not p1.exists() or not p2.exists(): die("文件不存在")
    diff_cnt = cmp_mask(p1, p2, 0xFFF0)
    if diff_cnt == 0:
        print("✅ 全部一致")
    else:
        print(f"❌ 共 {diff_cnt} / {p1.stat().st_size//2} 个 uint16 不匹配")
