import os
from dataclasses import dataclass

import torch
from vllm.attention.ops.flashmla import get_mla_metadata
from vllm.v1.attention.backends.mla.common import (
    MLACommonDecodeMetadata,
    MLACommonMetadataBuilder,
)
from vllm.v1.attention.backends.mla.flashmla import FlashMLAMetadata

from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse


@dataclass
class FlashMLADecodeMetadata(MLACommonDecodeMetadata):
    tile_scheduler_metadata: torch.Tensor
    num_splits: torch.Tensor
    topk_seq_lens: torch.Tensor
    topk_tile_scheduler_metadata: torch.Tensor
    topk_num_splits: torch.Tensor
    topk_block_table: torch.Tensor = None


class FlashMLAMetadataBuilder(MLACommonMetadataBuilder[FlashMLAMetadata]):

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
    ) -> FlashMLADecodeMetadata:
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            seq_lens_device,
            self.num_q_heads,
            1,  # MQA for the decode path
        )
        topk_seq_lens = None
        topk_tile_scheduler_metadata = None
        topk_num_splits = None
        if has_ucm_sparse():
            ucm_sparse = get_ucm_sparse()
            if os.getenv("VLLM_HASH_ATTENTION") == "1":
                topk_seq_lens, topk_tile_scheduler_metadata, topk_num_splits = (
                    ucm_sparse.build_decode_hash(seq_lens_device)
                )

        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            assert self.cg_buf_tile_scheduler_metadata is not None
            assert self.cg_buf_num_splits is not None

            sm_parts = tile_scheduler_metadata.size(0)
            # Metadata per-SM, upper bound on size (<= #SMs, TileMetadataSize)
            assert sm_parts <= self.cg_buf_tile_scheduler_metadata.size(0)
            tile_scheduler_metadata_view = self.cg_buf_tile_scheduler_metadata[
                :sm_parts
            ]
            tile_scheduler_metadata_view.copy_(tile_scheduler_metadata)
            tile_scheduler_metadata = tile_scheduler_metadata_view

            # Num splits is per-batch, varying size (batch_size,)
            n = num_splits.size(0)
            # make sure static buffer is large enough
            assert n <= self.cg_buf_num_splits.size(0)
            num_splits_view = self.cg_buf_num_splits[:n]
            num_splits_view.copy_(num_splits)
            # Num splits needs to monotonically increasing
            # (with: https://github.com/vllm-project/FlashMLA/pull/3, otherwise
            #  it needs to monotonically increasing by 1)
            self.cg_buf_num_splits[n:].fill_(num_splits[-1])
            num_splits = num_splits_view
            if has_ucm_sparse():
                ucm_sparse = get_ucm_sparse()
                if os.getenv("VLLM_HASH_ATTENTION") == "1":
                    topk_tile_scheduler_metadata, topk_num_splits, topk_seq_lens = (
                        ucm_sparse.maybe_init_cudagraph_buffers_for_topk(
                            n,
                            tile_scheduler_metadata,
                            topk_tile_scheduler_metadata,
                            topk_num_splits,
                            topk_seq_lens,
                        )
                    )
        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            topk_seq_lens=topk_seq_lens,
            topk_tile_scheduler_metadata=topk_tile_scheduler_metadata,
            topk_num_splits=topk_num_splits,
        )
