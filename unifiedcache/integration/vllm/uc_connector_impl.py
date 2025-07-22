from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import hash_request_tokens
from vllm.v1.core.sched.output import SchedulerOutput

from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector.base import Task
from unifiedcache.ucm_connector.factory import UcmConnectorFactory

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class LoadPara:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int = 0
    # Number of tokens cached in ssd
    storage_cached_tokens: int = 0
    # Whether the scheduler allow us to load the tokens
    can_load: bool = False
    # block hashes
    block_hashes: list[str] = field(default_factory=list)


@dataclass
class SavePara:
    # dump block ids
    num_blocks_need_save: int = 0
    # start save position
    start_save_position: int = 0
    # block hashes
    block_hashes: list[str] = field(default_factory=list)
    # num blocks prepare to save
    num_blocks_to_save: int = 0
    # num blocks already saved
    num_blocks_saved: int = 0


@dataclass
class ReqMeta:
    # Request ID, unique for each request
    request_id: str
    # Request block id in vllm
    vllm_block_ids: list[int]
    # Load information
    load_paras: Optional[LoadPara] = None
    # Save information
    save_paras: Optional[SavePara] = None


@dataclass
class UCConnectorV1Metadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        requests = []
    
    def add_request(self, request_id: str,
                    vllm_block_ids: list[int],
                    load_paras: Optional[LoadPara] = None,
                    save_paras: Optional[SavePara] = None
                    ) -> None:
        self.requests.append(
            ReqMeta(
                request_id=request_id,
                vllm_block_ids=vllm_block_ids,
                load_paras=load_paras,
                save_paras=save_paras))


class UCConnectorImpl:

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.load_paras: dict[str, LoadPara] = {}
        self.save_paras: dict[str, SavePara] = {}
        self.use_layerwise = True
        if self.vllm_config.kv_transfer_config is not None and \
                "ucm_connector_name" in self.vllm_config.kv_transfer_config.kv_connector_extra_config:
            name = self.vllm_config.kv_transfer_config.kv_connector_extra_config["ucm_connector_name"]
            config = None
            if "ucm_connector_config" in self.vllm_config.kv_transfer_config.kv_connector_extra_config:
                config = self.vllm_config.kv_transfer_config.kv_connector_extra_config["ucm_connector_config"]
            logger.info("init UCConnectorImpl, connector: %s", name)
            self.connector = UcmConnectorFactory.create_connector(name, config)
        else:
            raise TypeError(f"no storage connector")
        if self.vllm_config.kv_transfer_config is not None and \
                "use_layerwise" in self.vllm_config.kv_transfer_config.kv_connector_extra_config:
            self.use_layerwise = self.vllm_config.kv_transfer_config.kv_connector_extra_config["use_layerwise"]
    
    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        # TODO
        block_ids = []
        dst_tensor = []
        offset = []
        self.connector.load(block_ids, dst_tensor, offset)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # TODO
        self.connector.wait(Task())

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                        attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        # TODO
        block_ids = []
        dst_tensor = []
        offset = []
        self.connector.dump(block_ids, dst_tensor, offset)

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        # TODO
        self.connector.wait(Task())

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
            self,
            request: "Request",
            num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        assert num_computed_tokens % self.block_size == 0
        block_hash_types = hash_request_tokens(sha256,
                                               self.block_size, request)
        block_hashes: List[str] = [str(x.hash_value) for x in block_hash_types]
        if not block_hashes:
            logger.debug("Maybe tokens too short to load.")
            return 0, False
        hit_masks = self.connector.lookup(block_hashes)
        num_external_computed_tokens = sum(hit_masks) * self.block_size
        self.load_paras[request.request_id] = LoadPara(
            vllm_cached_tokens=num_computed_tokens,
            storage_cached_tokens=num_external_computed_tokens,
            block_hashes=block_hashes,
            can_load=False
        )

        num_max_cached_tokens = max(num_external_computed_tokens, num_computed_tokens)
        num_blocks_need_save = (len(request.all_tokens_ids) - num_max_cached_tokens) // self.block_size
        if num_blocks_need_save > 0:
            start_save_position = num_max_cached_tokens // self.block_size
            need_allocate_block_hashes = block_hashes[start_save_position:]
            ret = self.connector.create(need_allocate_block_hashes)
            self.save_paras[request.request_id] = SavePara(
                num_blocks_need_save=num_blocks_need_save,
                start_save_position=start_save_position,
                block_hashes=need_allocate_block_hashes
            )

        logger.debug(
            f"num_blocks_need_save = {num_blocks_need_save},\n"
            f"num_external_computed_tokens = {num_external_computed_tokens},\n"
            f"num_computed_tokens = {num_computed_tokens}.\n"
        )

        return max(num_external_computed_tokens - num_computed_tokens, 0), False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if request.request_id not in self.load_paras:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens > 0:
            self.load_paras[request.request_id].can_load = True

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function shoulf NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = UCConnectorV1Metadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            # Load kv is only supported for new reqs
            new_scheduled_blocks = scheduler_output.num_scheduled_tokens[new_req.req_id] // self.block_size
            load_paras = self.load_paras.pop(new_req.req_id, None)
            save_paras = self.save_paras.get(new_req.req_id, None)
            if save_paras is not None:
                save_paras.num_blocks_to_save = new_scheduled_blocks
            meta.add_request(new_req.req_id,
                             vllm_block_ids=new_req.block_ids[0],
                             load_paras=load_paras,
                             save_paras=save_paras)

        # 针对chunk prefill场景 running队列中的request可能仍然需要save
        for cached_req in scheduler_output.scheduled_cached_reqs:
            if cached_req.resumed_from_preemption:
                continue

            save_paras = self.save_paras.get(new_req.req_id, None)
            if save_paras is None:
                continue
            save_paras.num_blocks_saved += save_paras.num_blocks_to_save
            if save_paras.num_blocks_need_save > save_paras.num_blocks_to_save:
                logger.info(f"Running request {cached_req.req_id} has blocks to save")
                save_paras.start_save_position = 0
                new_scheduled_blocks = scheduler_output.num_scheduled_tokens[cached_req.req_id] // self.block_size
                save_paras.num_blocks_to_save = new_scheduled_blocks
                meta.add_request(cached_req.req_id,
                                 vllm_block_ids=cached_req.new_block_ids[0],
                                 load_paras=None,
                                 save_paras=save_paras)
        return meta
