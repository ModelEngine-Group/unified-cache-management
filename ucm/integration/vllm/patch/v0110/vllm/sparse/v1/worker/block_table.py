class BlockTable:

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def reset_row(
        self,
        row_idx: int,
    ) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.block_table.gpu[row_idx].fill_(0)
        self.block_table.cpu[row_idx].fill_(0)
        self.block_table.np[row_idx].fill(0)


class MultiGroupBlockTable:

    def reset_row(self, row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.reset_row(row_idx)
