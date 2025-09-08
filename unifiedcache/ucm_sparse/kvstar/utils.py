import subprocess

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


def execute_command(cmd_list):
    with subprocess.Popen(cmd_list,
                          shell=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        out, err = p.communicate(timeout=1000)
    res = out.decode()
    return res

def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = execute_command(["lscpu"]).split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")

                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

            numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl

def bind_cpus(world_size, rank_id, ratio=0.5):
    # 假设
    devices = list(range(world_size))

    numa_nodes_num = 1
    keyword = "NUMAnode(s)"
    numa_info = execute_command(["lscpu"]).split("\n")
    for _ in numa_info:
        line = ''.join(_.split())
        if keyword not in line:
            continue
        numa_nodes_num = int(line[-1])
        break

    print(f"numa_nodes_num: {numa_nodes_num}")
    alloc_numa_num = numa_nodes_num // world_size
    alloc_numa_ids = [i for i in range(rank_id * alloc_numa_num, (rank_id + 1) * alloc_numa_num)]
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

