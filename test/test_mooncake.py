import uuid
import hashlib
import torch

from unifiedcache.ucm_connector.base import Task
from unifiedcache.ucm_connector.ucm_mooncake import UcmMooncakeStore
from unifiedcache.logger import init_logger

logger = init_logger(__name__)


def tensor_hash(tensor: torch.Tensor) -> str:
    """Calculate the hash value of the tensor."""
    tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
    hash_object = hashlib.blake2b(tensor_bytes)
    hash_hex = hash_object.hexdigest()
    return str(int(hash_hex[:16], 16))


def test_lookup_not_found():
    """Test that lookup returns False for non-existent block IDs."""
    store = UcmMooncakeStore()
    block_ids = [uuid.uuid4().hex for _ in range(10)]
    masks = store.lookup(block_ids)
    assert all(mask is False for mask in masks)


def test_lookup_found():
    """Test that lookup returns True for existing block IDs after dumping data."""
    src_block_data = [torch.randint(0, 1000, (1, 100), dtype=torch.int) for _ in range(5)]
    block_ids = [tensor_hash(data) for data in src_block_data]
    offset = [0] * len(block_ids)

    store = UcmMooncakeStore()
    task: Task = store.dump(block_ids=block_ids, offset=offset, src_tensor=src_block_data)
    ret = store.wait(task)
    assert ret == 0
    masks = store.lookup(block_ids)
    assert all(mask is True for mask in masks)


def test_dump_once():
    """Test dumping data once and verifying it exists in the store."""
    src_block_data = [torch.randint(0, 1000, (1, 100), dtype=torch.int) for _ in range(5)]
    block_ids = [tensor_hash(data) for data in src_block_data]
    offset = [0] * len(block_ids)

    store = UcmMooncakeStore()
    task: Task = store.dump(block_ids=block_ids, offset=offset, src_tensor=src_block_data)
    ret = store.wait(task)
    assert ret == 0
    masks = store.lookup(block_ids)
    assert all(mask is True for mask in masks)


def test_dump_repeated():
    """Test that repeated dumping of the same data doesn't cause errors."""
    src_block_data = [torch.randint(0, 1000, (1, 100), dtype=torch.int) for _ in range(5)]
    block_ids = [tensor_hash(data) for data in src_block_data]
    offset = [0] * len(block_ids)

    store = UcmMooncakeStore()
    task: Task = store.dump(block_ids=block_ids, offset=offset, src_tensor=src_block_data)
    ret = store.wait(task)
    assert ret == 0
    masks = store.lookup(block_ids)
    assert all(mask is True for mask in masks)

    task: Task = store.dump(block_ids=block_ids, offset=offset, src_tensor=src_block_data)
    ret = store.wait(task)
    assert ret == 0


def test_load_existing_data():
    """Test loading data that was previously dumped into the store."""
    src_block_data = [torch.randint(0, 1000, (1, 100), dtype=torch.int) for _ in range(5)]
    dst_block_data = [torch.empty(data.shape, dtype=data.dtype) for data in src_block_data]
    block_ids = [tensor_hash(data) for data in src_block_data]
    offset = [0] * len(block_ids)

    store = UcmMooncakeStore()
    task: Task = store.dump(block_ids=block_ids, offset=offset, src_tensor=src_block_data)
    ret = store.wait(task)
    assert ret == 0

    masks = store.lookup(block_ids)
    assert all(mask is True for mask in masks)

    task: Task = store.load(block_ids=block_ids, offset=offset, dst_tensor=dst_block_data)
    ret = store.wait(task)
    assert ret == 0
    assert all([torch.equal(src_block_data[i], dst_block_data[i]) is True for i in range(len(src_block_data))])


def test_load_non_existent_data():
    """Test loading data that doesn't exist in the store verifies the destination remains unchanged."""
    src_block_data = [torch.randint(0, 1000, (1, 100), dtype=torch.int) for _ in range(5)]
    dst_block_data = [torch.empty(data.shape, dtype=data.dtype) for data in src_block_data]
    block_ids = [tensor_hash(data) for data in src_block_data]
    offset = [0] * len(block_ids)
    store = UcmMooncakeStore()
    masks = store.lookup(block_ids)
    assert all(mask is False for mask in masks)

    task: Task = store.load(block_ids=block_ids, offset=offset, dst_tensor=dst_block_data)
    ret = store.wait(task)
    assert ret != 0
    assert all([torch.equal(src_block_data[i], dst_block_data[i]) is False for i in range(len(src_block_data))])
