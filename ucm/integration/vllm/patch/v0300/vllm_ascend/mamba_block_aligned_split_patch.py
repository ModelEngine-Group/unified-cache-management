# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve checkpoint alignment during Ascend 0.30.0 chunked prefill."""

from functools import wraps

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.patch.platform.patch_mamba_block_aligned_split")
def patch_mamba_block_aligned_split(mod):
    # Patch after Ascend registers its wrapper on Scheduler, otherwise Ascend
    # would overwrite UCM's replacement during its own module import.
    original = mod._mamba_block_aligned_split
    if getattr(original, "_ucm_prefill_alignment_patched", False):
        return

    @wraps(original)
    def mamba_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        kv_transfer_config = self.vllm_config.kv_transfer_config
        # A computed prefix may just be an earlier chunk of a cold prefill.
        # Keep aligning while more than one real token remains, including replay
        # of a preempted request's output. Only preserve the consumer verifier
        # window once at most the final real token is left (or during decode).
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        remaining_tokens = (
            max(request.num_prompt_tokens, request.num_tokens) - num_computed_tokens
        )
        if (
            kv_transfer_config is not None
            and kv_transfer_config.is_kv_consumer
            and num_computed_tokens > 0
            and remaining_tokens <= 1
        ):
            return num_new_tokens

        # Pure PD prefill producer or standalone instance: suppress the EAGLE
        # one-block backoff (see the docstring). The sparse index-kpool
        # early-return path below must observe it too (it previously ran under
        # a temporary bit clear from the outer producer wrapper).
        if mod._get_sparse_index_kpool(self.vllm_config.model_config) is not None:
            num_computed_tokens = (
                request.num_computed_tokens
                + num_new_local_computed_tokens
                + num_external_computed_tokens
            )
            if num_computed_tokens < max(
                request.num_prompt_tokens,
                request.num_tokens - 1,
            ):
                block_size = self.block_size
                last_cache_position = (
                    request.num_tokens - request.num_tokens % block_size
                )
                if self.use_eagle and not mod._skips_eagle_block_drop(
                    kv_transfer_config
                ):
                    last_cache_position = max(last_cache_position - block_size, 0)
                scheduled_end = num_computed_tokens + num_new_tokens
                if scheduled_end < last_cache_position:
                    chunked_tokens = num_new_tokens // block_size * block_size
                    if chunked_tokens > 0:
                        num_new_tokens = chunked_tokens
                elif num_computed_tokens < last_cache_position < scheduled_end:
                    num_new_tokens = last_cache_position - num_computed_tokens
            return num_new_tokens

        if not mod._skips_eagle_block_drop(kv_transfer_config):
            return mod._original_mamba_block_aligned_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        # vLLM 0.28.x and newer revisions use different names.  Some
        # transitional scheduler implementations expose both and different
        # wrapper layers consult different attributes, so clear every
        # attribute that exists and restore all of them after the call.
        drop_attrs = tuple(
            name
            for name in ("use_eagle", "use_eagle_block_drop")
            if hasattr(self, name)
        )
        original_drop_values = {name: getattr(self, name) for name in drop_attrs}
        for name in drop_attrs:
            setattr(self, name, False)
        try:
            return mod._original_mamba_block_aligned_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        finally:
            for name, value in original_drop_values.items():
                setattr(self, name, value)

    mamba_block_aligned_split._ucm_prefill_alignment_patched = True
    patch_or_inject(mod, "_mamba_block_aligned_split", mamba_block_aligned_split)
    patch_or_inject(
        mod.Scheduler, "_mamba_block_aligned_split", mamba_block_aligned_split
    )
    logger.info("UCM Ascend 0.30.0 chunked-prefill alignment patch applied")
