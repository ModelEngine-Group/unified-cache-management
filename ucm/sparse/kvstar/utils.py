import collections
import hashlib
import pickle
import subprocess
from functools import cache


@cache
def get_offset(block_shape, rank, tp_size, precision, layer_id, is_v, is_mla) -> int:
    block_size, num_key_heads_per_tp, head_size = block_shape
    k_min_data_block_size = block_size * num_key_heads_per_tp * head_size * precision
    v_min_data_block_size = k_min_data_block_size if not is_mla else 0
    layer_size = (k_min_data_block_size + v_min_data_block_size) * tp_size
    if is_mla:
        k_offset = layer_size * layer_id
    else:
        k_offset = layer_size * layer_id + layer_size // tp_size * rank
    v_offset = k_offset + k_min_data_block_size
    return v_offset if is_v else k_offset


@cache
def compute_layer_offset(
    block_data_size: int,
    layer_id: int,
    is_v: bool,
    is_mla: bool,
) -> int:
    layer_data_size = block_data_size if is_mla else block_data_size * 2

    k_offset = layer_data_size * layer_id

    if is_mla:
        return k_offset

    v_offset = k_offset + block_data_size
    return v_offset if is_v else k_offset


@cache
def md5(input) -> int:
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    md5_bytes = hashlib.md5(input_bytes).digest()
    return int.from_bytes(md5_bytes, byteorder="big")


@cache
def block_hash_func(parent_block_hash, curr_block_token_ids):
    if not parent_block_hash:
        parent_block_hash = md5("UCMHASHSEED")
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return md5((parent_block_hash, curr_block_token_ids_tuple))


@cache
def compute_parent_block_hash(model_name, world_size, dtype, seed_rank=0) -> int:
    meta = f"{model_name}:{world_size}:{dtype}:{seed_rank}"
    meta_bytes = meta.encode("utf-8")
    h_seed = hashlib.md5(meta_bytes + b"UCM_HASH_SEED").digest()
    return int.from_bytes(h_seed, byteorder="big")


def execute_command(cmd_list):
    with subprocess.Popen(
        cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        out, err = p.communicate(timeout=1000)
    res = out.decode()
    return res


def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = execute_command(["lscpu"]).split("\n")
    for _ in cpu_info:
        line = "".join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")

                ranges += [
                    cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)
                ]

            numa_id = int(split_info[0].replace(keyword1, "").replace(keyword2, ""))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


def bind_cpus(world_size, rank_id, ratio=0.5):
    # 假设
    devices = list(range(world_size))

    numa_nodes_num = 1
    keyword = "NUMAnode(s)"
    numa_info = execute_command(["lscpu"]).split("\n")
    for _ in numa_info:
        line = "".join(_.split())
        if keyword not in line:
            continue
        numa_nodes_num = int(line[-1])
        break

    print(f"numa_nodes_num: {numa_nodes_num}")
    alloc_numa_num = numa_nodes_num // world_size
    alloc_numa_ids = [
        i for i in range(rank_id * alloc_numa_num, (rank_id + 1) * alloc_numa_num)
    ]
    print(f"alloc_numa_ids: {alloc_numa_ids}")
    cpu_idx_tbl = _get_cpu_info(alloc_numa_ids)
    print(f"cpu_idx_tbl: {cpu_idx_tbl}")

    phy_cpu_core_per_numa = 1
    for k in cpu_idx_tbl.keys():
        phy_cpu_core_per_numa = len(cpu_idx_tbl[k])
        break

    cpu_core_alloc = {}
    for numa in cpu_idx_tbl.keys():
        core_num = int(len(cpu_idx_tbl[numa]) * ratio)
        cpu_core_alloc[numa] = cpu_idx_tbl[numa][:core_num]

    print(f"cpu_core_alloc: {cpu_core_alloc}")

    return numa_nodes_num, alloc_numa_ids, phy_cpu_core_per_numa


def get_physical_core_topology():
    """
    use lscpu -e parse accurate cpu topology
    return a dict, key: numa_id, value: physical core ids in this numa
    """
    # topology[numa_id][core_id] = logical_cpu_id
    # make sure each physical core only record once
    topology = collections.defaultdict(dict)

    # execute lscpu -e, split as line
    # e.g.: 36  0    0      0    0:0:0:0       yes    3700.0000 1000.0000
    lscpu_output = execute_command(["lscpu", "-e"]).strip().split("\n")

    # skip title
    for line in lscpu_output[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue

        logical_cpu_id = int(parts[0])
        numa_id = int(parts[1])
        core_id = int(parts[3])  # physical core id

        if core_id not in topology[numa_id]:
            topology[numa_id][core_id] = logical_cpu_id

    final_mapping = {
        numa_id: list(sorted(cores.values())) for numa_id, cores in topology.items()
    }
    return final_mapping


def get_bind_cpus_for_rank(world_size, rank_id, ratio=1.0):
    """
    for each rank, compute alloc numa id

    scenario:
    1. numa_num >= world_size, equal division numa for each rank
    2. numa_num < world_size, equal division total cores for each rank
    """
    physical_core_map = get_physical_core_topology()
    if not physical_core_map:
        print("Could not determine CPU topology. Aborting bind.")
        return [], []

    print(f"Detected Physical Core Topology: {physical_core_map}")

    numa_nodes_num = len(physical_core_map)
    sorted_numa_ids = sorted(physical_core_map.keys())

    bind_info_list = []
    alloc_numa_ids = []

    numas_per_rank = numa_nodes_num // world_size

    if numas_per_rank > 0:
        print(f"Strategy: NUMA-level discard binding.")

        discarded_numa_count = numa_nodes_num % world_size
        if discarded_numa_count > 0:
            print(
                f"Note: {discarded_numa_count} NUMA node(s) (IDs: {sorted_numa_ids[-discarded_numa_count:]}) will be unused to ensure fair distribution."
            )

        start_numa_idx = rank_id * numas_per_rank
        end_numa_idx = start_numa_idx + numas_per_rank

        alloc_numa_ids = sorted_numa_ids[start_numa_idx:end_numa_idx]

        print(f"Rank {rank_id} allocated to NUMA nodes: {alloc_numa_ids}")

        for numa_id in alloc_numa_ids:
            physical_cores_on_numa = physical_core_map.get(numa_id, [])
            cores_to_take = int(len(physical_cores_on_numa) * ratio)
            for core_id in physical_cores_on_numa[:cores_to_take]:
                bind_info_list.append((core_id, numa_id))

    else:
        print(
            f"Strategy: Fallback to uniform core distribution ({world_size} ranks > {numa_nodes_num} NUMA nodes)."
        )

        all_physical_cores_with_numa = []
        for numa_id in sorted_numa_ids:
            for core_id in physical_core_map[numa_id]:
                all_physical_cores_with_numa.append((core_id, numa_id))

        total_physical_cores = len(all_physical_cores_with_numa)
        cores_per_rank = total_physical_cores // world_size
        if cores_per_rank == 0:
            print(
                f"Warning: Not enough physical cores ({total_physical_cores}) to assign at least one to each of the {world_size} ranks. Rank {rank_id} will not be bound to any core."
            )
            return [], sorted_numa_ids

        start_core_idx = rank_id * cores_per_rank
        end_core_idx = start_core_idx + cores_per_rank

        rank_core_share = all_physical_cores_with_numa[start_core_idx:end_core_idx]
        cores_to_take = int(len(rank_core_share) * ratio)
        bind_info_list = rank_core_share[:cores_to_take]

        alloc_numa_ids = sorted_numa_ids

    bind_info_list.sort()
    print(
        f"Rank {rank_id} will bind to {len(bind_info_list)} (CPU, NUMA) pairs: {bind_info_list}"
    )
    return bind_info_list, alloc_numa_ids
