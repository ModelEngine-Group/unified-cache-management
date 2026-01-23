from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

if hasattr(torch, "npu") and torch.npu.is_available():
    import torch_npu
    import ucm_custom_ops
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

from vllm import _custom_ops as ops
from vllm.attention.ops.flashmla import get_mla_metadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.utils import cdiv
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseCpuGpuBuffer,
    UcmSparseMetadata,
    UcmSparseRole,
)

if hasattr(torch, "cuda") and torch.cuda.is_available():
    from ucm.sparse.gsa_on_device.hamming_topk import (
        cuda_hamming_topk,
        fake_hamming_topk,
    )
    from ucm.sparse.gsa_on_device.hash_encoder import reshape_and_cache_khash_triton

from vllm.utils import cdiv

from ucm.sparse.gsa_on_device.gsa_on_device_config import GSAOnDeviceConfig
from ucm.sparse.gsa_on_device.hash_encoder import HashEncoder
from ucm.utils import Config

logger = init_logger(__name__)

ReqType = Union[str, int]


def gsa_on_device_config_path_for_model(vllm_config) -> str:
    model = vllm_config.model_config.model.lower()
    logger.info("[GSAOnDevice] model name: %s", model)

    if "deepseek" in model and "r1" in model:
        rel = (
            "ucm/sparse/gsa_on_device/configs/gsa_on_device_deepseek_r1_awq_config.json"
        )
    elif "qwen3" in model and "32b" in model and "coder" not in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_32B_config.json"
    elif "qwen3" in model and "30b" in model and "coder" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_coder_30B_A3B_config.json"
    elif "qwen3" in model and "4b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_4B_config.json"
    elif "qwq" in model and "32b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwq_32B_config.json"
    elif "deepseek" in model and "v2" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_deepseek_v2_lite_config.json"
    else:
        raise ValueError(f"[GSAOnDevice] Unsupported model for gsa_on_device: {model}")

    logger.info("[GSAOnDevice] target relative path: %s", rel)

    cur = Path(__file__).resolve()
    repo = cur
    for depth in range(30):
        if (
            (repo / "pyproject.toml").is_file()
            or (repo / "setup.cfg").is_file()
            or (repo / ".git").exists()
        ):

            p = repo / rel
            logger.info("[GSAOnDevice] repo root detected at depth=%d: %s", depth, repo)
            if p.is_file():
                logger.info("[GSAOnDevice] config loaded from SOURCE tree: %s", p)
                return str(p)
            logger.warning("[GSAOnDevice] repo root found but config missing: %s", p)
            break
        if repo.parent == repo:
            logger.debug("[GSAOnDevice] reached filesystem root, stop searching")
            break

        repo = repo.parent

    sub = rel[len("ucm/") :] if rel.startswith("ucm/") else rel
    res = resources.files("ucm").joinpath(*sub.split("/"))

    with resources.as_file(res) as p:
        logger.info("[GSAOnDevice] config loaded from PACKAGE resource (wheel): %s", p)
        return str(p)


class GSAOnDevice(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.rank = vllm_config.parallel_config.rank
        self.is_mla = vllm_config.model_config.is_deepseek_mla

        if vllm_config.device_config.device_type == "cuda":
            self.is_cuda = True
            self.device = torch.device(f"cuda:{self.rank}")
        elif vllm_config.device_config.device_type == "npu":
            self.is_cuda = False
            self.device = torch.device(f"npu:{self.rank}")
        else:
            raise ValueError(
                f"Unsupported device type: {vllm_config.device_config.device_type}"
            )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.block_size = vllm_config.cache_config.block_size

        # auto detect config file for GSAOnDevice
        gsa_on_device_config_path = gsa_on_device_config_path_for_model(vllm_config)

        self.gsa_on_device_config = GSAOnDeviceConfig.from_json(
            gsa_on_device_config_path
        )
        logger.info(f"read gsa_on_device config file : {gsa_on_device_config_path} ")

        self.hash_topk_tokens = self.gsa_on_device_config.vllm_hash_attention_topk
        self.hash_rollback_layers = (
            self.gsa_on_device_config.vllm_hash_attention_rollback_layers
        )
        self.hash_skip_layers = (
            self.gsa_on_device_config.vllm_hash_attention_skip_layers
        )

        self.seq_len_threshhold = self.gsa_on_device_config.seq_len_threshhold

        assert (
            self.seq_len_threshhold
            >= self.gsa_on_device_config.vllm_hash_attention_topk
        ), "seq_len_threshhold must be larger than or equal to vllm_hash_attention_topk"
        assert (
            self.gsa_on_device_config.vllm_hash_attention_topk % self.block_size == 0
        ), "vllm_hash_attention_topk must be divisible by block_size"
        assert (
            self.gsa_on_device_config.vllm_hash_attention_topk
            <= vllm_config.model_config.max_model_len
        ), "vllm_hash_attention_topk must be less than max_model_len"

        if role == UcmSparseRole.WORKER:
            if self.is_cuda:  # cuda only variables
                if not vllm_config.model_config.enforce_eager:
                    device_properties = torch.cuda.get_device_properties(self.device)
                    num_sms = device_properties.multi_processor_count
                    self.cg_buf_topk_tile_scheduler_metadata = torch.zeros(
                        (num_sms, 8),
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.cg_buf_topk_num_splits = torch.empty(
                        (vllm_config.scheduler_config.max_num_seqs + 1),
                        device=self.device,
                        dtype=torch.int32,
                    )

            self.ori_seq_lens_decode = None
            self.ori_block_table_decode = None
            self.origin_tile_scheduler_metadata = None  # for MLA
            self.origin_num_splits = None  # for MLA

            # for GQA
            self.topk_block_table = None
            self.topk_seq_lens = None
            self.topk_seq_lens_qwen = None
            self.has_pc_hit = False

            self.cached_reqs_to_step: dict[str, int] = dict()

            self._k_scale = torch.tensor(1.0, dtype=torch.float32)

            if self.is_mla:
                logger.info("GSAOnDevice initialized with MLA model config")
                self.hash_reduction_head_num = (
                    self.gsa_on_device_config.vllm_hash_attention_reduction_head_num
                )
                self.kv_lora_rank = getattr(
                    vllm_config.model_config.hf_text_config, "kv_lora_rank", None
                )
                self.qk_rope_head_dim = getattr(
                    vllm_config.model_config.hf_text_config, "qk_rope_head_dim", None
                )
                self.hash_encoder_nope = HashEncoder(
                    input_dim=self.kv_lora_rank,
                    hash_bits=self.kv_lora_rank,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )

                self.hash_encoder_rope = HashEncoder(
                    input_dim=self.qk_rope_head_dim,
                    hash_bits=self.qk_rope_head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )
            else:  # for GQA
                logger.info("GSAOnDevice initialized with non-MLA model config")
                self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
                self.max_num_tokens = vllm_config.model_config.max_model_len
                self.decode_req_ids_buf = self._make_buffer(
                    self.max_batch_size, dtype=torch.int64
                )

                self.init_for_pc()

                self.head_dim = vllm_config.model_config.get_head_size()
                self.hash_encoder = HashEncoder(
                    input_dim=self.head_dim,
                    hash_bits=self.head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )
                self.has_decode = False
                self.decode_only = False

                if not self.is_cuda:  # NPU only variables
                    self.decode_mask_npu = None
                    self.is_tensor_computed = False

                    self.hamming_keep_chunks_head = 1
                    self.hamming_keep_chunks_tail = 4

                    self.chunk_sizes_for_hamming_full = torch.full(
                        [self.max_batch_size],
                        fill_value=self.block_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.topk_for_hamming_full = torch.full(
                        [self.max_batch_size],
                        fill_value=self.hash_topk_tokens // self.block_size,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.topk_for_hamming_full_cpu = torch.full(
                        [self.max_batch_size],
                        fill_value=self.hash_topk_tokens // self.block_size,
                        dtype=torch.int32,
                        device="cpu",
                    )
                    self.seq_lens_for_hamming = torch.zeros(
                        [self.max_batch_size], dtype=torch.int32, device=self.device
                    )
                    self.hamming_output = torch.zeros(
                        [
                            self.max_batch_size,
                            self.num_key_heads,
                            cdiv(
                                vllm_config.model_config.max_model_len, self.block_size
                            ),
                        ],
                        dtype=torch.int32,
                        device=self.device,
                    )

    def init_for_pc(self):
        # for pc hit
        self.prefix_slot_mapping_buf = torch.empty(
            self.max_num_tokens * self.max_batch_size,
            device=self.device,
            dtype=torch.int64,
        )
        self.prefix_block_ids_buf = torch.empty(
            cdiv(self.max_num_tokens * self.max_batch_size, self.block_size),
            device=self.device,
            dtype=torch.int32,
        )
        self.token_idx_buf = torch.arange(
            self.max_num_tokens, device=self.device, dtype=torch.int64
        )

    def _make_buffer(
        self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True
    ) -> UcmSparseCpuGpuBuffer:
        return UcmSparseCpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    def hash_code(
        self,
        nope: Optional[torch.Tensor] = None,
        rope: Optional[torch.Tensor] = None,
        reduction_head_num: int = 1,
        query: Optional[torch.Tensor] = None,
    ):
        if self.is_mla:
            if nope is None or rope is None:
                raise ValueError("MLA mode requires `nope` and `rope`.")
            if reduction_head_num > 1:
                # reduce heads: [T, H, D] -> [T, H/reduce, D]
                nope = nope.view(
                    nope.shape[0],
                    reduction_head_num,
                    nope.shape[1] // reduction_head_num,
                    nope.shape[2],
                ).mean(dim=1)
                rope = rope.view(
                    rope.shape[0],
                    reduction_head_num,
                    rope.shape[1] // reduction_head_num,
                    rope.shape[2],
                ).mean(dim=1)
            hash_nope = self.hash_encoder_nope.compute_hash(nope)
            hash_rope = self.hash_encoder_rope.compute_hash(rope)
            return hash_nope.view(torch.bfloat16), hash_rope.view(torch.bfloat16)

        # ---- GQA mode ----
        else:
            if query is None:
                raise ValueError("GQA mode requires `query`.")
            if self.num_q_heads > self.num_key_heads:
                query = query.view(
                    query.shape[0],
                    self.num_key_heads,
                    self.num_q_heads // self.num_key_heads,
                    query.shape[2],
                ).mean(2)
            elif self.num_q_heads < self.num_key_heads:
                query = torch.repeat_interleave(
                    query, self.num_key_heads // self.num_q_heads, dim=1
                )

            return self.hash_encoder.compute_hash(query).view(torch.bfloat16)

    def get_layer_attn_metadata(self, forward_context: ForwardContext, layer_name: str):
        attn_meta = forward_context.attn_metadata
        return attn_meta[layer_name] if isinstance(attn_meta, dict) else attn_meta

    def get_layer_state(self, layer_name: str):
        layer_id = int(layer_name.split(".")[2])
        is_rollback_layer = layer_id in self.hash_rollback_layers
        is_skip_hash_layer = (
            layer_id < len(self.hash_skip_layers) and self.hash_skip_layers[layer_id]
        )
        return is_rollback_layer, is_skip_hash_layer

    def cache_k_hash_gqa_cuda(
        self, key, attn_metadata, k_hash, forward_context, layer_name
    ):
        k_hash_compute = self.hash_encoder.compute_hash(key).view(torch.bfloat16)
        valid_k_hash_token = attn_metadata.slot_mapping.flatten().numel()
        reshape_and_cache_khash_triton(
            k_hash_compute[:valid_k_hash_token],
            attn_metadata.slot_mapping.flatten(),
            k_hash,
            block_size=self.block_size,
        )
        if self.has_pc_hit:
            ## 重新捞取所有token的key
            attn = forward_context.no_compile_layers[layer_name]
            kv_cache = attn.kv_cache[forward_context.virtual_engine]

            k_cache = kv_cache[0][0][self.prefix_block_ids]
            k_cache = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
            prefix_k_hash_compute = self.hash_encoder.compute_hash(k_cache).view(
                torch.bfloat16
            )
            prefix_valid_k_hash_token = self.prefix_slot_mapping.flatten().numel()
            reshape_and_cache_khash_triton(
                prefix_k_hash_compute[:prefix_valid_k_hash_token],
                self.prefix_slot_mapping.flatten(),
                k_hash,
                block_size=self.block_size,
            )

    def cache_k_hash_gqa_npu(self, key, k_hash, attn_metadata):
        if not self.is_tensor_computed:
            if self.decode_mask.any():  # with at least one decode request
                decode_req_ids = torch.nonzero(
                    self.decode_mask, as_tuple=False
                ).flatten()
                decode_req_ids_npu = torch.nonzero(
                    self.decode_mask_npu, as_tuple=False
                ).flatten()
                batch_size_for_hamming = self.decode_mask.sum().item()
                self.query_lens_device = attn_metadata.query_lens_device[
                    decode_req_ids_npu
                ]
                self.topk_for_hamming = self.topk_for_hamming_full[
                    :batch_size_for_hamming
                ]
                self.chunk_sizes_for_hamming = self.chunk_sizes_for_hamming_full[
                    :batch_size_for_hamming
                ]
                self.seq_lens_for_hamming = attn_metadata.seq_lens_device[
                    decode_req_ids_npu
                ]
                self.max_seq_len_for_hamming = torch.max(
                    attn_metadata.seq_lens[decode_req_ids]
                ).item()
                self.is_tensor_computed = True

        k_hash_compute = self.hash_encoder.compute_hash(key)
        assert (
            k_hash_compute.shape[0] == attn_metadata.slot_mapping.numel()
        ), f"shape mismatch: k_hash_compute.shape[0]={k_hash_compute.shape[0]} != attn_metadata.slot_mapping.numel()={attn_metadata.slot_mapping.numel()}"
        k_hash_compute = (
            k_hash_compute.transpose(0, 1)
            .reshape(-1, k_hash_compute.shape[-1])
            .contiguous()
        )
        ucm_custom_ops.reshape_and_cache_bnsd(
            k_hash_compute,
            k_hash,
            attn_metadata.slot_mapping,
            attn_metadata.query_lens_device,  # need to modify attention_v1.py in vllm-asecnd
            k_hash,
        )

    def update_decode_topk_mla(
        self,
        is_rollback_layer,
        is_skip_hash_layer,
        attn_metadata,
        decode_ql_nope,
        decode_q_pe,
        k_hash,
    ):
        if not is_rollback_layer:
            if is_skip_hash_layer:
                assert attn_metadata.decode.topk_block_table is not None
                block_table = attn_metadata.decode.topk_block_table
            else:
                q_nope_hash, q_rope_hash = self.hash_code(
                    nope=decode_ql_nope,
                    rope=decode_q_pe,
                    reduction_head_num=self.hash_reduction_head_num,
                )
                q_hash = torch.cat([q_nope_hash, q_rope_hash], dim=-1)
                topk_token = self.hash_topk_tokens
                block_table = cuda_hamming_topk(
                    q_hash.unsqueeze(1),
                    k_hash.unsqueeze(2),
                    attn_metadata.decode.block_table,
                    attn_metadata.decode.seq_lens,
                    topk_token=topk_token,
                    sink_token=64,
                    recent_token=512,
                    is_mla=self.is_mla,
                )
                attn_metadata.decode.topk_block_table = block_table

            seq_lens = attn_metadata.decode.topk_seq_lens
            tile_scheduler_metadata = attn_metadata.decode.topk_tile_scheduler_metadata
            num_splits = attn_metadata.decode.topk_num_splits

            self.ori_block_table_decode = attn_metadata.decode.block_table
            self.ori_seq_lens_decode = attn_metadata.decode.seq_lens
            self.origin_tile_scheduler_metadata = (
                attn_metadata.decode.tile_scheduler_metadata
            )
            self.origin_num_splits = attn_metadata.decode.num_splits

            attn_metadata.decode.block_table = block_table
            attn_metadata.decode.seq_lens = seq_lens
            attn_metadata.decode.tile_scheduler_metadata = tile_scheduler_metadata
            attn_metadata.decode.num_splits = num_splits

    def update_decode_topk_gqa_cuda(self, query, k_hash, attn_metadata):
        if self.decode_only:
            q_decode = query[: self.num_reqs]
        else:
            q_decode = query.index_select(0, self.decode_req_ids)
        q_hash = self.hash_code(query=q_decode)

        block_table_decode = cuda_hamming_topk(
            q_hash.unsqueeze(1),
            k_hash,
            self.block_table_decode,
            self.seq_len_decode,
            topk_token=self.hash_topk_tokens,
            sink_token=64,
            recent_token=512,
            is_mla=self.is_mla,
        )
        # update topk_block_table
        topk = block_table_decode.shape[1]
        if self.decode_only:
            # 直接 slice 写入
            self.new_block_table[: self.num_reqs, :topk] = block_table_decode
            self.new_block_table[: self.num_reqs, topk:] = 0
            attn_metadata.block_table = self.new_block_table
            # update seq_lens
            self.new_seq_lens[: self.num_reqs] = self.topk_seq_lens_qwen
            attn_metadata.seq_lens = self.new_seq_lens
        else:
            self.new_block_table[self.decode_req_ids, :topk] = block_table_decode
            self.new_block_table[self.decode_req_ids, topk:] = 0
            attn_metadata.block_table = self.new_block_table

            # update seq_lens
            self.new_seq_lens.index_copy_(
                0, self.decode_req_ids, self.topk_seq_lens_qwen
            )
            attn_metadata.seq_lens = self.new_seq_lens

    def update_decode_topk_gqa_npu(self, query, k_hash, attn_metadata):
        q_start = attn_metadata.query_start_loc
        decode_req_ids = torch.nonzero(self.decode_mask_npu, as_tuple=False).flatten()
        decode_token_idx = q_start[:-1].index_select(0, decode_req_ids)
        q_decode = query.index_select(0, decode_token_idx)

        q_hash = self.hash_encoder.compute_hash(q_decode).unsqueeze(2).contiguous()

        block_table_decode = attn_metadata.block_table.index_select(0, decode_req_ids)

        ucm_custom_ops.hamming_dist_top_k(
            q_hash,
            k_hash,
            self.topk_for_hamming,
            self.seq_lens_for_hamming,
            self.chunk_sizes_for_hamming,
            self.max_seq_len_for_hamming,
            self.hamming_keep_chunks_head,
            self.hamming_keep_chunks_tail,
            0,  # support_offload is disabled
            block_table_decode,
            self.hamming_output[: len(decode_req_ids)],
        )
        topk = self.hamming_output.shape[-1]
        attn_metadata.block_table[decode_req_ids, :topk] = self.hamming_output[
            : len(decode_req_ids), 0, :
        ]
        attn_metadata.block_table[decode_req_ids, topk:] = 0

        # we have already computed the topk_seq_lens_qwen in `build_decode_attention_meta_npu()`
        attn_metadata.seq_lens[self.decode_mask] = self.topk_seq_lens_qwen

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
        # TODO: Should mark MTP layer as rollback layer
        is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)

        if not is_rollback_layer and not is_skip_hash_layer:
            if self.is_mla:
                k_c_normed_hash, k_pe_hash = self.hash_code(nope=key, rope=value)
                ops.concat_and_cache_mla(
                    k_c_normed_hash,
                    k_pe_hash.squeeze(1),
                    k_hash,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype="auto",
                    scale=self._k_scale,
                )
                # external_pc_hit need fix
            else:  # GQA
                if self.is_cuda:
                    self.cache_k_hash_gqa_cuda(
                        key, attn_metadata, k_hash, forward_context, layer_name
                    )
                else:  # NPU
                    self.cache_k_hash_gqa_npu(key, k_hash, attn_metadata)
        if self.is_mla:
            if phase == "decode":
                self.update_decode_topk_mla(
                    is_rollback_layer,
                    is_skip_hash_layer,
                    attn_metadata,
                    decode_ql_nope,
                    decode_q_pe,
                    k_hash,
                )
        else:  # GQA
            if self.has_decode:  # 有decode阶段的req
                if not is_rollback_layer:
                    if is_skip_hash_layer:
                        # 跳层 使用上一个topk结果
                        if self.is_cuda:
                            attn_metadata.block_table = self.topk_block_table
                        else:
                            attn_metadata.block_tables = self.topk_block_table
                        attn_metadata.seq_lens = self.topk_seq_lens
                    else:
                        if self.is_cuda:
                            self.update_decode_topk_gqa_cuda(
                                query, k_hash, attn_metadata
                            )
                        else:  # NPU
                            self.update_decode_topk_gqa_npu(
                                query, k_hash, attn_metadata
                            )
                        # topk for skip layer
                        self.topk_block_table = attn_metadata.block_table
                        self.topk_seq_lens = attn_metadata.seq_lens

        return query, key, value, output

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
        if self.is_mla:
            if phase == "decode":
                # TODO: Should mark MTP layer as rollback layer
                is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
                if not is_rollback_layer:
                    attn_metadata.decode.block_table = self.ori_block_table_decode
                    attn_metadata.decode.seq_lens = self.ori_seq_lens_decode
                    attn_metadata.decode.tile_scheduler_metadata = (
                        self.origin_tile_scheduler_metadata
                    )
                    attn_metadata.decode.num_splits = self.origin_num_splits
        else:  # 判断req decode阶段
            if self.has_decode:
                if self.is_cuda:
                    attn_metadata.block_table = self.ori_block_table_decode
                else:
                    attn_metadata.block_tables = self.ori_block_table_decode
                attn_metadata.seq_lens = self.ori_seq_lens_decode

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def execute_begin(self, scheduler_output: SchedulerOutput):
        self.is_tensor_computed = False

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        return INVALID_SLOT

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        dtype = torch.bfloat16
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache_shape = list(
                    (kv_cache if self.is_mla else kv_cache[0]).shape
                )
                khash_cache_shape[-1] //= dtype.itemsize * 8
                khash_cache = torch.zeros(khash_cache_shape, dtype=dtype, device=device)
            else:
                khash_cache = None
            kv_caches[layer_name] = (kv_cache, khash_cache)

    def initialize_kv_hash_cache_tensors_npu(self, kv_caches, device):
        print(
            f"[NPU GSAOnDevice Debug] initialize_kv_hash_cache_tensors_npu: allocating hashk cache for GSAOnDevice in NPU"
        )
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            k_cache_shape = kv_cache[0].shape
            print(
                f"[NPU GSAOnDevice Debug] layer_name: {layer_name}, is_rollback_layer={is_rollback_layer}, is_skip_hash_layer={is_skip_hash_layer}, k_cache_shape: {k_cache_shape}"
            )
            khash_cache_shape = (
                k_cache_shape[0],
                k_cache_shape[2],
                k_cache_shape[1],
                self.hash_encoder.hash_bits // 8,
            )
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache = torch.empty(
                    khash_cache_shape, dtype=torch.uint8, device=device
                )
                print(
                    f"[NPU GSAOnDevice Debug] layer_name: {layer_name}, khash_cache_shape: {khash_cache_shape}"
                )
            else:
                khash_cache = None
                print(
                    f"[NPU GSAOnDevice Debug] layer_name: {layer_name}, khash_cache is None"
                )
            kv_caches[layer_name] = (kv_cache, khash_cache)

    def build_decode_hash(self, seq_lens):
        from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

        topk_seq_lens = update_seq_lens(
            seq_lens,
            topk_token=self.hash_topk_tokens,
            block_size=self.block_size,
        )
        topk_tile_scheduler_metadata, topk_num_splits = get_mla_metadata(
            topk_seq_lens,
            self.num_q_heads,
            1,
        )
        return topk_seq_lens, topk_tile_scheduler_metadata, topk_num_splits

    def rebuild_prefix_cache_info_for_req(
        self,
        block_table_row: torch.Tensor,
        num_prompt_tokens: int,
        qlen: int,
        block_size: int,
    ):
        """
        num_prefix_tokens: 命中 prefix 的 token 数
        prefix_block_ids:  命中 prefix 覆盖的 block ids（block-level）
        prefix_slot_mapping: 命中 prefix 的 slot ids（token-level）
        """
        assert 0 <= qlen <= num_prompt_tokens
        num_prefix_tokens = num_prompt_tokens - qlen
        if num_prefix_tokens <= 0:
            empty = block_table_row[:0]
            return 0, 0, empty, empty

        num_prefix_blocks = (num_prefix_tokens + block_size - 1) // block_size
        prefix_block_ids = block_table_row[:num_prefix_blocks]  # [prefix_blocks]

        token_idx = self.token_idx_buf[:num_prefix_tokens]
        block_indices = token_idx // block_size
        block_offsets = token_idx - block_indices * block_size
        token_block_numbers = prefix_block_ids.index_select(0, block_indices)

        prefix_slot_mapping = token_block_numbers * block_size + block_offsets
        return (
            num_prefix_tokens,
            num_prefix_blocks,
            prefix_block_ids,
            prefix_slot_mapping,
        )

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:
        from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

        if not self.is_mla:
            self.has_decode = False
            self.decode_only = False
            self.has_pc_hit = False

            num_decodes = 0
            # for pc
            num_pc_hit = 0
            all_prefix_tokens = 0
            all_prefix_blocks = 0

            if isinstance(attn_metadata, dict):
                attn_metadata = next(iter(attn_metadata.values()))

            compute_q_lens = (
                attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
            )
            self.decode_req_ids_buf.clear()

            self.num_reqs = len(scheduler_output.num_scheduled_tokens)
            for (
                req_id,
                num_scheduled_tokens,
            ) in scheduler_output.num_scheduled_tokens.items():
                req = requests[req_id]
                # req_state: is_decode  is_first_prefil is_prefill is_last_chunk
                is_decode = (
                    req_id in self.cached_reqs_to_step
                    and self.cached_reqs_to_step[req_id]
                    > 0  # step always=0 when prefill
                )
                is_first_prefil = (
                    req_id not in self.cached_reqs_to_step
                )  # first prefill when chunkprefill
                is_prefill = is_first_prefil or self.cached_reqs_to_step[req_id] == 0
                is_last_chunk = is_prefill and (
                    req.num_computed_tokens + num_scheduled_tokens
                    >= req.num_prompt_tokens
                )

                # when prompt length < topk_tokens Skip sparse!
                if req.num_prompt_tokens < self.hash_topk_tokens:
                    continue

                if is_decode:
                    self.decode_req_ids_buf.np[num_decodes] = (
                        input_batch.req_id_to_index[req_id]
                    )
                    num_decodes += 1

                if is_first_prefil:
                    self.cached_reqs_to_step[req_id] = 0
                    # num_prompt_tokens -> store pc -> rebuild slotmapping
                    req_row_id = input_batch.req_id_to_index[req_id]
                    ext_tokens = int(
                        scheduler_output.num_external_computed_tokens_per_req.get(
                            req_id, 0
                        )
                    )
                    if ext_tokens > 0:
                        (
                            num_prefix_tokens,
                            num_prefix_blocks,
                            prefix_block_ids,
                            prefix_slot_mapping,
                        ) = self.rebuild_prefix_cache_info_for_req(
                            block_table_row=attn_metadata.block_table[req_row_id],
                            num_prompt_tokens=req.num_prompt_tokens,
                            qlen=compute_q_lens[req_row_id],
                            block_size=self.block_size,
                        )

                        self.prefix_slot_mapping_buf[
                            all_prefix_tokens : all_prefix_tokens + num_prefix_tokens
                        ] = prefix_slot_mapping
                        self.prefix_block_ids_buf[
                            all_prefix_blocks : all_prefix_blocks + num_prefix_blocks
                        ] = prefix_block_ids

                        all_prefix_tokens += num_prefix_tokens
                        all_prefix_blocks += num_prefix_blocks
                        num_pc_hit += 1

                if is_last_chunk:
                    self.cached_reqs_to_step[req_id] += 1

            self.has_decode = num_decodes > 0
            self.decode_only = self.has_decode and (num_decodes == self.num_reqs)
            if self.has_decode:
                # for roll_back recode the full seqlens & block_table
                self.ori_seq_lens_decode = attn_metadata.seq_lens.clone()
                self.ori_block_table_decode = attn_metadata.block_table.clone()

                if self.decode_only:
                    decode_seq_lens = self.ori_seq_lens_decode[: self.num_reqs]
                    self.block_table_decode = self.ori_block_table_decode[
                        : self.num_reqs
                    ]
                    self.seq_len_decode = self.ori_seq_lens_decode[: self.num_reqs]
                else:
                    self.decode_req_ids_buf.copy_to_gpu(num_decodes)
                    self.decode_req_ids = self.decode_req_ids_buf.gpu[:num_decodes]
                    decode_seq_lens = attn_metadata.seq_lens.index_select(
                        0, self.decode_req_ids
                    )

                    self.block_table_decode = attn_metadata.block_table.index_select(
                        0, self.decode_req_ids
                    )
                    self.seq_len_decode = attn_metadata.seq_lens.index_select(
                        0, self.decode_req_ids
                    )

                self.topk_seq_lens_qwen = update_seq_lens(
                    decode_seq_lens,
                    topk_token=self.hash_topk_tokens,
                    block_size=self.block_size,
                )

                self.new_block_table = attn_metadata.block_table
                self.new_seq_lens = attn_metadata.seq_lens

            self.has_pc_hit = num_pc_hit > 0
            if self.has_pc_hit:
                self.prefix_slot_mapping = self.prefix_slot_mapping_buf[
                    :all_prefix_tokens
                ]
                self.prefix_block_ids = self.prefix_block_ids_buf[:all_prefix_blocks]

    def maybe_init_cudagraph_buffers_for_topk(self, n, tile_scheduler_metadata):
        sm_parts = tile_scheduler_metadata.size(0)
        topk_tile_scheduler_metadata_view = self.cg_buf_topk_tile_scheduler_metadata[
            :sm_parts
        ]
        topk_tile_scheduler_metadata_view.copy_(topk_tile_scheduler_metadata)
        topk_tile_scheduler_metadata = topk_tile_scheduler_metadata_view

        topk_num_splits_view = self.cg_buf_topk_num_splits[:n]
        topk_num_splits_view.copy_(topk_num_splits)
        self.cg_buf_topk_num_splits[n:].fill_(topk_num_splits[-1])
        topk_num_splits = topk_num_splits_view
        return topk_tile_scheduler_metadata, topk_num_splits

    def _free_cached_request(self, request_id: Union[int, str]) -> None:
        if request_id not in self.cached_reqs_to_step:
            return
        del self.cached_reqs_to_step[request_id]

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self._free_cached_request(req_id)

        req_data = scheduler_output.scheduled_cached_reqs
        for req_id, resumed_from_preemption in zip(
            req_data.req_ids, req_data.resumed_from_preemption
        ):
            if resumed_from_preemption:
                self._free_cached_request(req_id)
