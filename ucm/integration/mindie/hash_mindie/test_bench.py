import random
import time

import numpy as np
import uc_hash_ext

HASH_SHIFT_LEFT = 6
HASH_SHIFT_RIGHT = 2
INVALID_HASH_VALUE = 0
EXTRA_HASH = INVALID_HASH_VALUE
CONST = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


def cpp_style_hash(value):
    if isinstance(value, int):
        return value
    else:
        if isinstance(value, str):
            hash_value = 0
            for char in value:
                hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFFFFFFFFFF
            return hash_value
    return hash(value)


def hash_combine(seed, token_id):
    seed ^= (
        cpp_style_hash(token_id)
        + CONST
        + (seed << HASH_SHIFT_LEFT)
        + (seed >> HASH_SHIFT_RIGHT)
    )
    seed = 1 if seed == INVALID_HASH_VALUE else seed
    return seed % 2**64


def hash_block_py(prefix_hash_value, block_token_ids):
    seed = INVALID_HASH_VALUE
    if prefix_hash_value != INVALID_HASH_VALUE:
        seed = hash_combine(seed, prefix_hash_value)
    for token_id in block_token_ids:
        seed = hash_combine(seed, token_id)
    seed = hash_combine(seed, EXTRA_HASH)
    return seed


def hash_prefix_py(tokens, block_size, start_block, end_block) -> np.ndarray:
    out = []
    prefix_hash_value = INVALID_HASH_VALUE
    for i in range(0, end_block):
        prefix_hash_value = hash_block_py(
            prefix_hash_value, tokens[i * block_size : (i + 1) * block_size]
        )
        out.append(prefix_hash_value)
    return np.asarray(out[start_block:], dtype=np.uint64)


def check_hash_prefix(iters=1000):
    random.seed(1)
    for _ in range(iters):
        block_size = random.choice([1, 4, 8, 16, 32, 64, 128])
        num_blocks = random.randint(0, 200)
        total = block_size * num_blocks

        toks = [random.randint(0, 200_000) for _ in range(total)]

        start_block = 0
        end_block = random.randint(start_block, num_blocks)

        for dtype in (np.uint64, np.int32, np.int64):
            arr = np.asarray(toks, dtype=dtype)

            py_out = hash_prefix_py(toks, block_size, start_block, end_block)
            cpp_out = uc_hash_ext.hash_prefix(
                0, arr, block_size, start_block, end_block
            )

            if py_out.shape != cpp_out.shape or not np.array_equal(py_out, cpp_out):
                print(
                    f"Test failed for block_size={block_size}, num_blocks={num_blocks}, dtype={dtype}"
                )
                print(f"Expected: {py_out}")
                print(f"Got: {cpp_out}")
                return False

    print("All tests passed!")
    return True


def bench():
    block_size = 128
    blocks = 128
    token_ids = np.arange(blocks * block_size, dtype=np.uint64)

    _ = uc_hash_ext.hash_prefix(0, token_ids, block_size, 0, blocks)

    seed = 0
    st = time.perf_counter()
    res_py = []
    for i in range(blocks):
        seed = hash_block_py(
            seed, token_ids[i * block_size : (i + 1) * block_size].tolist()
        )
        res_py.append(seed)
    t_py = time.perf_counter() - st

    seed = 0
    st = time.perf_counter()
    res_cpp_batch = uc_hash_ext.hash_prefix(0, token_ids, block_size, 0, blocks)
    t_cpp_batch = time.perf_counter() - st

    print(f"python loop + slice + tolist: {t_py:.6f} s")
    print(f"cpp hash_prefix batch: {t_cpp_batch:.6f} s")
    print("same[10]:", res_py[10] == res_cpp_batch[10])


if __name__ == "__main__":
    passed = check_hash_prefix()
    if passed:
        bench()
