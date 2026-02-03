import argparse
import glob
import math
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy.special import softmax


# Reuse-Aware Layer Skipping
class RALS:
    def __init__(self, args, reuse=False):
        self.args = args
        self.savepath = (
            f"./output/{os.path.basename(args.datapath)}/{args.calibration_data}"
        )
        path = Path(self.savepath)
        path.mkdir(parents=True, exist_ok=True)
        if reuse:
            with open(f"{self.savepath}/TK.pkl", "rb") as f:
                self.TK = pickle.load(f)
            with open(f"{self.savepath}/TQ.pkl", "rb") as f:
                self.TQ = pickle.load(f)
            with open(f"{self.savepath}/result.pkl", "rb") as f:
                self.RESULT = pickle.load(f)
            self.max_step = self.RESULT[0].shape[0] - 1
        else:
            self.KQloader()
            self.getSparsity()

    def extract_prefix(self, root, flag="_k_"):
        files = [f for f in glob.glob(f"{root}/*.pt") if flag in f]
        names = [os.path.basename(p) for p in files]
        return os.path.commonprefix(names)

    def load_tensor(self, path, dim):
        return torch.load(path, map_location="cpu").float().reshape(-1, dim).numpy()

    def KQloader(self):
        """
        K/Q Loading supports both:
            Standard Mode: KV heads sharded across TP, tensors organized per head/layer/timestep
            MLA Mode: Shared K, TP-sharded Q, dynamic decode-step expansion
        """
        num_groups = max(self.args.TP, self.args.kvheads)

        self.TK = {i: {} for i in range(num_groups)}
        self.TQ = {i: {} for i in range(num_groups)}

        root_base = f"{self.args.datapath}/{self.args.device}"

        # ======================================================
        # MLA 模式
        # ======================================================
        if self.args.enable_mla:
            root = f"{root_base}0/{self.args.calibration_data}"

            prefix_k = self.extract_prefix(root, "_k_")

            # ---------- 获取最大 decode step ----------
            base_dir = Path(root)
            files = base_dir.glob(
                f"{prefix_k}{self.args.layers - 1}.self_attn.attn_custom*.pt"
            )

            max_file = max(files, key=lambda p: int(p.stem.split("attn_custom")[-1]))
            self.max_step = int(max_file.stem.split("attn_custom")[-1])

            # ---------- 加载 K/Q ----------
            for layer in tqdm.tqdm(range(self.args.layers), desc="Loading MLA tensors"):
                # Load K (只在 TP0)
                k_path = (
                    f"{root}/{prefix_k}{layer}.self_attn.attn_custom{self.max_step}.pt"
                )
                K = self.load_tensor(k_path, self.args.dim)
                K = K[np.any(K != 0, axis=1)]  # 去零行
                self.TK[0][layer] = K

                # Load Q (所有 TP + 所有 step)
                for tp in range(self.args.TP):
                    tp_root = root.replace(
                        f"{self.args.device}0", f"{self.args.device}{tp}"
                    )
                    prefix_q = prefix_k.replace("_k_", "_q_")

                    q_steps = []
                    for step in range(self.max_step + 1):
                        q_path = (
                            f"{tp_root}/{prefix_q}{layer}"
                            f".self_attn.attn_custom{step}.pt"
                        )
                        q_steps.append(self.load_tensor(q_path, self.args.dim))

                    self.TQ[tp][layer] = q_steps
        else:
            # ======================================================
            # 非 MLA 模式
            # ======================================================
            root = f"{root_base}0/{self.args.calibration_data}"

            prefix_k = self.extract_prefix(root, "_k_")
            prefix_q = prefix_k.replace("_k_", "_q_")

            heads_per_tp = self.args.kvheads // self.args.TP

            for tp in range(self.args.TP):
                tp_root = root.replace(
                    f"{self.args.device}0", f"{self.args.device}{tp}"
                )

                for layer in tqdm.tqdm(range(self.args.layers), desc=f"TP{tp}"):
                    # ---------- Load K ----------
                    k_path = f"{tp_root}/{prefix_k}{layer}.self_attn.pt"
                    k_dict = torch.load(k_path)

                    K_list = [v for k, v in k_dict.items() if k[0] > 0]
                    K = torch.cat(K_list, dim=0).float().cpu().numpy()
                    K = K.reshape(K.shape[0], -1, self.args.dim)

                    # ---------- Load Q ----------
                    q_path = f"{tp_root}/{prefix_q}{layer}.self_attn.pt"
                    q_dict = torch.load(q_path)

                    Q_list = [
                        v for k, v in q_dict.items() if k[0] > 0 and v.shape[0] == 1
                    ]
                    Q = torch.cat(Q_list, dim=0).float().cpu().numpy()
                    Q = Q.reshape(Q.shape[0], -1, self.args.dim)

                    # ---------- 拆 kv head ----------
                    for h in range(heads_per_tp):
                        idx = tp * heads_per_tp + h

                        self.TK[idx][layer] = K[:, h : h + 1, :]
                        self.TQ[idx][layer] = Q[
                            :, h * self.args.qhead : (h + 1) * self.args.qhead, :
                        ]

            self.max_step = (
                self.TQ[self.args.kvheads - 1][self.args.layers - 1].shape[0] - 1
            )

        with open(f"{self.savepath}/TK.pkl", "wb") as f:
            pickle.dump(self.TK, f)
        with open(f"{self.savepath}/TQ.pkl", "wb") as f:
            pickle.dump(self.TQ, f)

    def aggregate_chunks(self, score, chunk_size):
        pad = (-len(score)) % chunk_size
        if pad:
            score = np.pad(score, (0, pad))
        return score.reshape(-1, chunk_size).sum(axis=1)

    def compute_ratio90(self, score, chunk_size):
        score_rep = self.aggregate_chunks(score, chunk_size)
        v = np.sort(score_rep)[::-1]
        v /= v.sum()

        return np.searchsorted(np.cumsum(v), 0.9, side="right") / len(v)

    def getSparsity(self):
        """
        Hash Ratio sparsity measures how many attention blocks are required
        to cover 90% of total attention energy, enabling temporal sparsity
        profiling and rollback layer detection
        """
        self.RESULT = {}

        is_mla = self.args.enable_mla
        kv_range = [0] if is_mla else range(self.args.kvheads)

        for kvhead in kv_range:

            if not is_mla:
                print(f"Processing kvhead {kvhead}...")

            HISTORY = []

            for step in tqdm.tqdm(range(self.max_step + 1)):
                layer_ratios = []

                for layer in range(self.args.layers):
                    # ---------- Load K ----------
                    k = self.TK[0][layer] if is_mla else self.TK[kvhead][layer]
                    k = k[: k.shape[0] - self.max_step + step]

                    if is_mla:
                        k = k.reshape(k.shape[0], -1, self.args.dim)

                    # ---------- Load Q ----------
                    if is_mla:
                        q = np.concatenate(
                            [self.TQ[tp][layer][step] for tp in range(self.args.TP)],
                            axis=0,
                        )
                    else:
                        q = self.TQ[kvhead][layer][step]

                    # ---------- Attention score ----------
                    score = np.einsum("bij,kj->kb", k, q) * self.args.scale
                    score = softmax(score, axis=1).sum(axis=0)

                    # ---------- Sparsity ratio ----------
                    ratio90 = self.compute_ratio90(score, self.args.chunk_size)
                    layer_ratios.append(ratio90)

                HISTORY.append(layer_ratios)

            self.RESULT[kvhead] = np.asarray(HISTORY)

        with open(f"{self.savepath}/result.pkl", "wb") as f:
            pickle.dump(self.RESULT, f)

    def remove_outliers_iqr(self, data):
        q25 = np.quantile(data, 0.25, axis=0)
        q75 = np.quantile(data, 0.75, axis=0)
        iqr = q75 - q25

        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        return np.clip(data, lower, upper)

    def plot_history(self, history, mean_curve, save_path, title):
        plt.figure(figsize=(8, 3))

        for row in history:
            plt.plot(row, alpha=0.5)

        plt.plot(mean_curve, color="black", marker="s", label="mean", linewidth=2)

        plt.xlabel("Layer")
        plt.ylabel("Hash Ratio")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def calRollBackLayers(self, sparsity):
        kvhead_means = []

        # =====================================================
        # Per-kvhead statistics & plots
        # =====================================================
        for kvhead in range(self.args.kvheads):
            history = self.RESULT[kvhead]

            # ---------- IQR outlier removal ----------
            history_clean = self.remove_outliers_iqr(history)

            # ---------- Mean curve ----------
            mean_curve = history_clean.mean(axis=0)
            kvhead_means.append(mean_curve)

            # ---------- Plot ----------
            self.plot_history(
                history_clean,
                mean_curve,
                Path(self.savepath) / f"hratio_{kvhead}.png",
                title=f"Block-level (kvhead {kvhead})",
            )

        # =====================================================
        # Overall rollback decision
        # =====================================================
        stat_overall = np.mean(kvhead_means, axis=0)

        rollback_k = math.ceil(self.args.layers * (1 - sparsity))

        # layers with highest hash ratio
        rollback_layers = np.argsort(stat_overall)[-rollback_k:]
        rollback_layers = np.sort(rollback_layers)

        # ---------- Overall plot ----------
        plt.figure(figsize=(8, 3))
        plt.plot(stat_overall, marker="s", color="black", label="overall_mean")
        plt.scatter(
            rollback_layers,
            stat_overall[rollback_layers],
            color="red",
            marker="s",
            label="rollback layers",
        )
        plt.xlabel("Layer")
        plt.ylabel("Hash Ratio")
        plt.title("Block-level Overall")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(self.savepath) / "hratio_overall.png", dpi=300)
        plt.close()

        print(f"Rollback layers (sparsity={sparsity}): {rollback_layers.tolist()}")

        return rollback_layers.tolist()

    def getoverlapMatrix(self, perserved_tokens):
        """
        The Attention Overlap Matrix captures inter-layer similarity by measuring
        top-k attention block intersections across layers.
        """
        first_loop = self.args.TP if self.args.enable_mla else self.args.kvheads

        for head in range(first_loop):

            print(f"Processing head {head}...")

            # overlap accumulator
            overlap_sum = np.zeros((self.args.layers, self.args.layers))
            count = self.max_step + 1

            for step in tqdm.tqdm(range(count)):

                topk_masks = []

                for layer in range(self.args.layers):
                    # ----------- Load K -----------
                    k = (
                        self.TK[0][layer]
                        if self.args.enable_mla
                        else self.TK[head][layer]
                    )
                    k = k[: k.shape[0] - self.max_step + step]
                    k = k.reshape(k.shape[0], -1, self.args.dim)

                    # ----------- Load Q -----------
                    q = self.TQ[head][layer][step]

                    scores = np.einsum("bij,kj->kb", k, q) * self.args.scale
                    attn = softmax(scores, axis=1).mean(axis=0)

                    score_rep = self.aggregate_chunks(attn, self.args.chunk_size)

                    # ----------- Mask sink & recent -----------
                    sink_chunks = self.args.sink // self.args.chunk_size
                    recent_chunks = self.args.recent // self.args.chunk_size

                    if sink_chunks > 0:
                        score_rep[:sink_chunks] = np.inf
                    if recent_chunks > 0:
                        score_rep[-recent_chunks:] = np.inf

                    topk = perserved_tokens // self.args.chunk_size

                    mask = np.zeros_like(score_rep, dtype=bool)
                    mask[np.argpartition(-score_rep, topk)[:topk]] = True
                    topk_masks.append(mask)

                # ----------- Overlap accumulation -----------
                for i in range(self.args.layers):
                    mi = topk_masks[i]
                    for j in range(i + 1, self.args.layers):
                        mj = topk_masks[j]
                        overlap_sum[i, j] += np.sum(mi & mj) / np.sum(mj)

            Table = overlap_sum / count
            np.fill_diagonal(Table, 1.0)

            # ----------- Plot heatmap (upper triangle only) -----------
            mask_lower = np.tril(np.ones_like(Table, dtype=bool))
            heat = np.where(mask_lower, np.nan, Table)

            plt.figure(figsize=(18, 12))
            plt.imshow(heat.T, cmap="Blues", interpolation="nearest")
            plt.colorbar()
            plt.title("Top-k Overlap Heatmap")
            plt.tight_layout()
            plt.savefig(
                Path(self.savepath) / f"top{perserved_tokens}_overlap_{head}.png",
                dpi=300,
            )
            plt.close()

            # ----------- Save matrix -----------
            np.save(Path(self.savepath) / f"matrix_{head}.npy", Table)

    def best_path_min_sum(self, A_list, max_changes):
        n = len(A_list[0])
        k = len(A_list)

        def value(i, j):
            return [A[i][j] for A in A_list]

        NEG_INF = -1e18

        # dp[j][i][c] = (min_val, sum_val, prev_i)
        dp = [[[None] * (max_changes + 1) for _ in range(n)] for _ in range(n)]

        # 初始化
        vals0 = value(0, 0)
        dp[0][0][0] = (min(vals0), sum(vals0), None)

        for j in range(1, n):
            for i in range(j + 1):
                vals = value(i, j)
                sum_vals = sum(vals)
                min_vals = min(vals)
                for c in range(max_changes + 1):
                    best = None
                    best_prev = None

                    # 不换行
                    if dp[j - 1][i][c] is not None:
                        prev_min, prev_sum, _ = dp[j - 1][i][c]
                        new_min = min(prev_min, min_vals)
                        new_sum = prev_sum + sum_vals
                        if best is None or (new_min, new_sum) > (best[0], best[1]):
                            best = (new_min, new_sum)
                            best_prev = i

                    # 换行 (只能跳到 i=j)
                    if i == j and c > 0:
                        for k in range(j):
                            if dp[j - 1][k][c - 1] is None:
                                continue
                            prev_min, prev_sum, _ = dp[j - 1][k][c - 1]
                            new_min = min(prev_min, min_vals)
                            new_sum = prev_sum + sum_vals
                            if best is None or (new_min, new_sum) > (best[0], best[1]):
                                best = (new_min, new_sum)
                                best_prev = k

                    if best is not None:
                        dp[j][i][c] = (*best, best_prev)

        # 找终点最优路径
        best_state = None
        best_i = best_c = None
        j = n - 1
        for i in range(n):
            for c in range(max_changes + 1):
                if dp[j][i][c] is None:
                    continue
                min_val, sum_val, _ = dp[j][i][c]
                if best_state is None or (min_val, sum_val) > best_state:
                    best_state = (min_val, sum_val)
                    best_i = i
                    best_c = c

        if best_i is None:
            return None, None, []

        # 回溯路径
        path = []
        cur_i, cur_c = best_i, best_c
        cur_j = n - 1
        while cur_j >= 0:
            path.append((cur_i, cur_j))
            _, _, prev_i = dp[cur_j][cur_i][cur_c]
            if cur_i == cur_j and cur_c > 0:
                cur_c -= 1
            cur_i = prev_i
            cur_j -= 1

        path.reverse()
        return best_c, best_state, path

    def plot_upper_triangle(self, mat, save_path, title):
        mask = np.tril(np.ones_like(mat, dtype=bool))
        data = np.where(mask, np.nan, mat)

        plt.figure(figsize=(18, 12))
        plt.imshow(data, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def getSkipLayerConfig(
        self, rollbackLayers, sparsity, computed_ratio_of_hamming_layers
    ):
        """
        An optimal skip-layer path is searched to maximize attention reuse
        under overlap and transition constraints.
        """
        base_dir = Path(f"./output/{os.path.basename(self.args.datapath)}")

        num_heads = self.args.TP if self.args.enable_mla else self.args.kvheads

        TABLELIST = []

        # =====================================================
        # Load & merge overlap matrices
        # =====================================================
        for name in self.args.skipLayerinput:

            input_dir = base_dir / name
            tables = []

            for h in range(num_heads):
                Table = np.load(input_dir / f"matrix_{h}.npy")

                # zero out rollback layers
                Table[rollbackLayers, :] = 0
                Table[:, rollbackLayers] = 0
                np.fill_diagonal(Table, 1.0)

                tables.append(Table)

            tables = np.asarray(tables)

            merged = tables.mean(0) if self.args.enable_mla else tables.min(0)
            TABLELIST.append(merged)

            self.plot_upper_triangle(
                merged,
                base_dir / f"matrix_{name}.png",
                title=f"Overlap matrix ({name})",
            )

        # =====================================================
        # Solve optimal skipping path
        # =====================================================
        max_changes = math.ceil(
            self.args.layers * sparsity * computed_ratio_of_hamming_layers
        ) + len(rollbackLayers)

        _, _, path = self.best_path_min_sum(TABLELIST, max_changes=max_changes)

        # =====================================================
        # Generate configs per table
        # =====================================================
        for idx, Table in enumerate(TABLELIST):

            parent = {}
            segment_scores = {}

            for i, j in path:
                if i == j:
                    segment_scores[i] = [1.0]
                    parent[j] = i
                else:
                    parent[j] = i
                    segment_scores[i].append(Table[i, j])

            # ---------- Config vector ----------
            config = [
                -1 if parent.get(i, i) == i else parent[i]
                for i in range(self.args.layers)
            ]

            print(f"\nCONFIG {idx}:\n========")
            print(config)

            # ---------- Compute layers ----------
            values = np.concatenate(list(segment_scores.values()))
            compute_layers = np.where(values == 1)[0]
            compute_layers = sorted(list(set(compute_layers) - set(rollbackLayers)))

            hamming_count = len(compute_layers)

            print(f"\nSUMMARY {idx}:\n========")
            print("ROLLBACK LAYERS: ", rollbackLayers)
            print("COMPUTED HAMMING LAYERS: ", compute_layers)

            # ---------- Visualization ----------
            plt.figure(figsize=(10, 2))
            plt.plot(values, marker="s")
            plt.xlabel("Layer")
            plt.ylabel("Overlap Ratio")
            plt.title(
                f"Rollback={rollbackLayers}, " f"Computed_hammings={hamming_count}"
            )
            plt.tight_layout()
            plt.savefig(base_dir / f"result_final_{idx}.png", dpi=300)
            plt.close()

    def run(
        self,
        perserved_tokens,
        sparsity,
        computed_ratio_of_hamming_layers,
        update_tokens_for_recompute=True,
    ):
        if update_tokens_for_recompute:
            self.getoverlapMatrix(perserved_tokens)
        rollbackLayers = self.calRollBackLayers(sparsity)
        self.getSkipLayerConfig(
            rollbackLayers, sparsity, computed_ratio_of_hamming_layers
        )


class Config:
    device = "cuda"  # 推理设备名称，cuda/npu
    enable_mla = True  # 是否启用 MLA Attention
    TP = 8  # tensor parallel 数
    kvheads = 1  # KV head 总数
    qhead = 128  # 每 KV head对应 Q head 数
    chunk_size = 64  # block 粒度（token 聚合单位）
    layers = 61  # 模型层数
    dim = 576  # head hidden size
    if enable_mla:
        scale = 1 / np.sqrt(128 + 64) * (0.1 * np.log(40) + 1) ** 2
    else:
        scale = 1 / np.sqrt(dim)
    sink = 64  # 强制保留开头 token
    recent = 512  # 强制保留最近 token
    datapath = "./output/_profiling/DeepSeekR1"  # profiling 数据路径
    calibration_data = "dahailaozhen_openthink"  # 数据集名称
    skipLayerinput = ["dahailaozhen_openthink"]


if __name__ == "__main__":
    args = Config()
    ins = RALS(args, reuse=False)
    ins.run(
        perserved_tokens=2048,
        sparsity=0.91,
        computed_ratio_of_hamming_layers=0.3,
        update_tokens_for_recompute=True,
    )
