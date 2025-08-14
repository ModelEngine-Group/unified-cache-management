import os
import hashlib
import torch

from unifiedcache.ucm_connector.ucm_mooncake import UcmMooncakeStore, MooncakeTask
from unifiedcache.logger import init_logger

logger = init_logger(__name__)


def tensor_hash(tensor: torch.Tensor) -> str:
    """Calculate the hash value of the tensor."""
    tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
    hash_object = hashlib.blake2b(tensor_bytes)
    hash_hex = hash_object.hexdigest()
    return str(int(hash_hex[:16], 16))

store = UcmMooncakeStore()
src_block_data = [torch.randint(0, 1000, (1,10), dtype=torch.int) for _ in range(5)]
dst_block_data = [torch.empty(data.shape, dtype=data.dtype) for data in src_block_data]
block_ids = [tensor_hash(data) for data in src_block_data]

mask = store.lookup(block_ids)
logger.info(f"First lookup: {mask=}")

task: MooncakeTask = store.dump(block_ids=block_ids, offset=[], src_tensor=src_block_data)
store.wait(task)
logger.info(f"Dump end: {task=}")

mask = store.lookup(block_ids)
logger.info(f"Second lookup: {mask=}")

task: MooncakeTask = store.load(block_ids=block_ids, offset=[], dst_tensor=dst_block_data)
store.wait(task)
logger.info(f"Load end: {task=}")

logger.info("原始张量Hash:")
logger.info(f"{block_ids=}")
logger.info("原始张量:")
logger.info(src_block_data)
logger.info("还原后的张量:")
logger.info(dst_block_data)
logger.info("是否一致:")
logger.info(f"{[torch.equal(src_block_data[i], dst_block_data[i]) for i in range(len(src_block_data))]}")
