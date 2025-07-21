from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Any, Generator

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import hash_request_tokens
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

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
    # Whether the scheduler allow us to load the blocks
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
    # num of blocks prepare to save
    num_blocks_to_save: int = 0
    # num of blocks already saved
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
        self.requests = []
    
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
        self.use_layerwise = True
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.rank = vllm_config.parallel_config.rank
        self.load_paras: dict[str, LoadPara] = {}
        self.save_paras: dict[str, SavePara] = {}
        self.dump_tasks: dict[str, List[Task]] = {}
        self.load_tasks: dict[str, tuple[Task, Task]] = {}
        self.failed_dump_requests: set[str] = set()
        self.is_mla = self.vllm_config.model_config.is_deepseek_mla
        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        if self.vllm_config.kv_transfer_config is not None and \
                "ucm_connector_name" in self.vllm_config.kv_transfer_config.kv_connector_extra_config:
            name = self.vllm_config.kv_transfer_config.kv_connector_extra_config["ucm_connector_name"]
            config = None
            if "ucm_connector_config" in self.vllm_config.kv_transfer_config.kv_connector_extra_config:
                config = self.vllm_config.kv_transfer_config.kv_connector_extra_config["ucm_connector_config"]
            logger.info("init UCConnectorImpl, connector: %s", name)
            self.connector = UcmConnectorFactory.create_connector(name, config)
        else:
            raise TypeError(f"no storage connector.")
        if self.vllm_config.kv_transfer_config is not None and \
                "use_layerwise" in self.vllm_config.kv_transfer_config.kv_connector_extra_config:
            self.use_layerwise = self.vllm_config.kv_transfer_config.kv_connector_extra_config["use_layerwise"]

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

    def DataLength(self, min_data_block_size, num_kv_heads):
        # 计算worker上一个block的kv = MinDataBlockSize * model.head_number / total_tp_size
        min_size = min_data_block_size * (num_kv_heads * self.total_tp_size)
        if self.is_mla:
            return min_size
        return min_size // self.total_tp_size

    def DataOffset(self, kv_layer, rank, layer_id, is_v):
        # 非MLA场景shape为 (2, num_blocks, block_size, num_kv_heads, head_size)
        # MLA场景shape为 (num_blocks, block_size, num_kv_heads, head_size)
        # TODO MLA适配
        kv_layer_shape = kv_layer.shape
        # 数据类型占的字节数
        elem_size = kv_layer.storage().element_size()
        num_kv_heads = kv_layer_shape[3] if not self.is_mla else kv_layer_shape[2]
        k_dim = kv_layer_shape[4] if not self.is_mla else kv_layer_shape[3]
        v_dim = k_dim if not self.is_mla else 0
        logger.debug(
            f"total_tp_size = {self.total_tp_size},\n"
            f"shape of layer = {kv_layer_shape},\n"
            f"num_kv_heads = {num_kv_heads},\n"
            f"k_dim = v_dim = head_size = {k_dim},\n"
            f"element size = {elem_size}."
        )
        # 一个head的k或v的偏移
        k_min_data_block_size = k_dim * self.block_size
        v_min_data_block_size = v_dim * self.block_size
        # layer_size = 一个head的kv * model.head_number
        # 多tp场景下 total_head_number = num_kv_heads * tp_size
        layer_size = (k_min_data_block_size + v_min_data_block_size) * (num_kv_heads * self.total_tp_size)
        if (is_v):
            # v的偏移 = k的偏移 + 一个block内k的偏移
            return int(
                self.DataOffset(kv_layer, rank, layer_id, False) + self.DataLength(k_min_data_block_size, num_kv_heads))
        if self.is_mla:
            return int(layer_size * layer_id)
        else:
            # k的偏移 = layer_size * 层数 + layer_size / 总tp * 当前tp
            return int(layer_size * layer_id + layer_size / self.total_tp_size * self.rank)

    def get_tensor_and_offset_layerwise(
            self,
            vllm_block_ids: List[int],
            kv_layer: torch.Tensor,
            layer_name: str) -> tuple[List[torch.Tensor], List[int]]:
        k_tensors = []
        k_offsets = []
        v_tensors = []
        v_offsets = []
        layer_id = self._extract_layer_index(layer_name)

        for blk_id in vllm_block_ids:
            k_layer_block = kv_layer[0][blk_id].contiguous()
            k_data_offset = self.DataOffset(kv_layer, self.rank, layer_id, False)
            k_tensors.append(k_layer_block)
            k_offsets.append(k_data_offset)
            if not self.is_mla:
                v_layer_block = kv_layer[1][blk_id].contiguous()
                v_data_offset = self.DataOffset(kv_layer, self.rank, layer_id, True)
                v_tensors.append(v_layer_block)
                v_offsets.append(v_data_offset)
        return k_tensors + v_tensors, k_offsets + v_offsets

    def generate_layerwise_load_tasks(self, fetch_block_hashes,
                                      layer_to_tensor: dict[
                                          str, tuple[List[torch.Tensor], List[int]]]) -> \
            Generator[
                Optional[tuple[Task, Task]], None, None]:

        logger.debug(f"fetch_block_hashes is {fetch_block_hashes}")
        assert fetch_block_hashes is not None and fetch_block_hashes, "The block hashes need to be fetched should not be None or empty."
        assert layer_to_tensor is not None and layer_to_tensor, "The layers of tensor need to be fetched should not be None or empty."

        blocks_len = len(fetch_block_hashes)
        def load(tensor_list, offset_list) -> tuple[Task, Task]:
            k_load_task = self.connector.load(fetch_block_hashes, offset_list[:blocks_len], tensor_list[:blocks_len])
            v_load_task = None
            if not self.is_mla:
                v_load_task = self.connector.load(fetch_block_hashes, offset_list[blocks_len:], tensor_list[blocks_len:])
            return k_load_task, v_load_task

        for layer_name, (tensor_list, offset_list) in layer_to_tensor.items():
            logger.debug(f"Start excute {layer_name} load task.")
            yield load(tensor_list, offset_list)

        yield None

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
        metadata = kwargs.get('metadata', None)
        assert isinstance(metadata, UCConnectorV1Metadata)

        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        self.layerwise_load_tasks: dict[str, Generator[tuple[Task, Task], None, None]] = {}
        self.current_layer = 0
        for request in metadata.requests:
            if request.load_paras is None or not request.load_paras.can_load:
                continue
            layer_to_tensor: dict[str, tuple[List[torch.Tensor], List[int]]] = {}
            block_ids = request.vllm_block_ids
            # 需要load的block id应从内存cached的block往后
            load_start_block_id = request.load_paras.vllm_cached_tokens // self.block_size
            load_end_block_id = request.load_paras.storage_cached_tokens // self.block_size
            fetch_block_ids = block_ids[load_start_block_id:load_end_block_id]
            logger.debug(
                f"fetch_block_ids = {fetch_block_ids},\n"
                f"load_start_block_id = {load_start_block_id},\n"
                f"load_end_block_id = {load_end_block_id},\n"
                f"fetch_block_ids = {fetch_block_ids}"
            )
            fetch_block_hashes = request.load_paras.block_hashes[load_start_block_id:load_end_block_id]
            assert len(fetch_block_ids) == len(fetch_block_hashes)
            blocks_len = len(fetch_block_ids)
            for layer_name, kv_layer in self.kv_caches.items():
                tensors, offsets = self.get_tensor_and_offset_layerwise(fetch_block_ids, kv_layer, layer_name)
                if not self.use_layerwise:
                    task = self.connector.load(fetch_block_hashes, offsets[:blocks_len], tensors[:blocks_len])
                    assert self.connector.wait(task) == 0
                    if not self.is_mla:
                        task = self.connector.load(fetch_block_hashes, offsets[blocks_len:], tensors[blocks_len:])
                        assert self.connector.wait(task) == 0
                else:
                    layer_to_tensor[layer_name] = (tensors, offsets)

            if layer_to_tensor:
                layerwise_load_task = self.generate_layerwise_load_tasks(fetch_block_hashes, layer_to_tensor)
                load_task = next(layerwise_load_task)
                assert load_task is not None, "The first layerwise task should not be None!"
                self.load_tasks[request.request_id] = load_task
                self.layerwise_load_tasks[request.request_id] = layerwise_load_task

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if not self.use_layerwise:
            return
        if self.layerwise_load_tasks:
            logger.info(f"Waiting for layer {self.current_layer} to be loaded")

        assert self.current_layer < self.num_layers, "The current layer should be less than total layers!"
        for request_id, gene_load_task in self.layerwise_load_tasks.items():
            k_task, v_task = self.load_tasks[request_id]
            assert self.connector.wait(k_task) == 0
            if v_task:
                assert self.connector.wait(v_task) == 0
            if self.current_layer < self.num_layers - 1:
                self.load_tasks[request_id] = next(gene_load_task)
                assert self.load_tasks[request_id] is not None, "The task for next layer should not be None!"
            else:
                logger.debug(f"Load tasks for {request_id} finished.")

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
        if not self.use_layerwise:
            return

        metadata = kwargs.get('metadata', None)
        assert isinstance(metadata, UCConnectorV1Metadata)
        assert attn_metadata is not None, "The attn_metadata should not be None."

        for request in metadata.requests:
            if request.save_paras is None:
                continue

            save_param = request.save_paras
            vllm_block_ids = request.vllm_block_ids[
                             save_param.start_save_position:save_param.start_save_position + save_param.num_blocks_to_save]
            blocks_len = len(vllm_block_ids)
            tensors, offsets = self.get_tensor_and_offset_layerwise(vllm_block_ids, kv_layer, layer_name)
            storage_block_ids = save_param.block_hashes[
                                save_param.num_blocks_saved:save_param.num_blocks_saved + save_param.num_blocks_to_save]
            logger.info(
                f"blocks length = {blocks_len},\n"
                f"length of offsets = {len(offsets)},\n"
                f"length of need save vllm_block_ids = {len(vllm_block_ids)},\n"
                f"length of storage_block_ids = {len(storage_block_ids)},\n"
            )
            task = self.connector.dump(storage_block_ids, offsets[:blocks_len], tensors[:blocks_len])
            self.dump_tasks.setdefault(request.request_id, []).append(task)
            if not self.is_mla:
                task = self.connector.dump(storage_block_ids, offsets[blocks_len:], tensors[blocks_len:])
                self.dump_tasks.setdefault(request.request_id, []).append(task)
        self.current_layer += 1

    def wait_for_save(self, metadata) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        assert isinstance(metadata, UCConnectorV1Metadata)
        if self.use_layerwise:
            for request_id, dump_tasks in self.dump_tasks.items():
                if any(self.connector.wait(task) != 0 for task in dump_tasks):
                    self.failed_dump_requests.add(request_id)
            # clear dump_tasks for all request
            self.dump_tasks.clear()
            return

        for request in metadata.requests:
            if request.save_paras is None:
                continue
            save_paras = request.save_paras
            logger.debug(
                f"num_blocks_saved = {save_paras.num_blocks_saved},\n"
                f"num_blocks_to_save = {save_paras.num_blocks_to_save}\n"
            )
            start_pos = save_paras.start_save_position
            num_blocks = save_paras.num_blocks_to_save
            num_blocks_saved = save_paras.num_blocks_saved

            dump_block_ids = request.vllm_block_ids[start_pos:start_pos + num_blocks]
            dump_vllm_block_hashes = save_paras.block_hashes[num_blocks_saved:num_blocks_saved + num_blocks]

            logger.debug(
                f"dump block ids is {dump_block_ids},\n"
                f"dump_vllm_block_hashes is {dump_vllm_block_hashes}\n"
            )

            assert len(dump_block_ids) == len(dump_vllm_block_hashes)
            blocks_len = len(dump_block_ids)
            tasks = []
            for layer_name, kv_layer in self.kv_caches.items():
                tensors, offsets = self.get_tensor_and_offset_layerwise(dump_block_ids, kv_layer, layer_name)
                task = self.connector.dump(dump_vllm_block_hashes, offsets[:blocks_len], tensors[:blocks_len])
                tasks.append(task)
                if not self.is_mla:
                    task = self.connector.dump(dump_vllm_block_hashes, offsets[blocks_len:], tensors[blocks_len:])
                    tasks.append(task)
            for task in tasks:
                if self.connector.wait(task) != 0:
                    self.failed_dump_requests.add(request.request_id)
                    break

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
        num_blocks_need_save = (len(request.all_token_ids) - num_max_cached_tokens) // self.block_size
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

        This function should NOT modify fields in the scheduler_output.
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
        # clear all load_paras when build meta for new reqs done
        self.load_paras.clear()

        # 针对chunk prefill场景 running队列中的request可能仍然需要save
        for cached_req in scheduler_output.scheduled_cached_reqs:
            if cached_req.resumed_from_preemption:
                continue

            save_paras = self.save_paras.get(cached_req.req_id, None)
            if save_paras is None:
                continue
            save_paras.num_blocks_saved += save_paras.num_blocks_to_save
            if save_paras.num_blocks_need_save > save_paras.num_blocks_saved:
                logger.debug(f"Running request {cached_req.req_id} has blocks to save")
                save_paras.start_save_position = 0
                new_scheduled_blocks = scheduler_output.num_scheduled_tokens[cached_req.req_id] // self.block_size
                save_paras.num_blocks_to_save = new_scheduled_blocks
                meta.add_request(cached_req.req_id,
                                 vllm_block_ids=cached_req.new_block_ids[0],
                                 load_paras=None,
                                 save_paras=save_paras)
        return meta

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        is_success = not (
                request.status == RequestStatus.FINISHED_ABORTED or request.request_id in self.failed_dump_requests)
        # clear save_paras for request
        save_paras = self.save_paras.pop(request.request_id, None)
        # clear load_tasks for request
        self.load_tasks.pop(request.request_id, None)
        # remove failed_dump_requests for request
        self.failed_dump_requests.discard(request.request_id)
        if save_paras:
            self.connector.commit(save_paras.block_hashes, is_success)
        return False, None

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None