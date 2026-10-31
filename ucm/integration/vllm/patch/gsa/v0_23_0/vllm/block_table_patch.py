"""Inject ``reset_row`` into vLLM 0.23.0 ``BlockTable`` / ``MultiGroupBlockTable``.

Replicates the ``vllm/v1/worker/block_table.py`` hunk of
``vllm-adapt-1020-c8.patch`` at runtime so no ``git apply`` is needed.
"""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _block_table_reset_row(self, row_idx: int) -> None:
    self.num_blocks_per_row[row_idx] = 0
    self.block_table.gpu[row_idx].fill_(0)
    self.block_table.cpu[row_idx].fill_(0)
    self.block_table.np[row_idx].fill(0)


def _multi_group_block_table_reset_row(self, row_idx: int) -> None:
    for _i, block_table in enumerate(self.block_tables):
        block_table.reset_row(row_idx)


@when_imported("vllm.v1.worker.block_table")
def _patch_vllm_block_table(mod):
    if not hasattr(mod, "BlockTable"):
        logger.warning("Skip vLLM block_table reset_row patch: BlockTable missing")
        return
    patch_or_inject(mod.BlockTable, "reset_row", _block_table_reset_row)
    if hasattr(mod, "MultiGroupBlockTable"):
        patch_or_inject(
            mod.MultiGroupBlockTable,
            "reset_row",
            _multi_group_block_table_reset_row,
        )
    logger.info("UCM GSA vLLM BlockTable.reset_row patch applied")
