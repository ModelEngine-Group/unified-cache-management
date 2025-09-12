def reset_row(self, row_idx: int,) -> None:
    self.num_blocks_per_row[row_idx] = 0
    self.block_table[row_idx].fill_(0)
    self.block_table_cpu[row_idx].fill_(0)
    self.block_table_np[row_idx].fill(0)

def reset_row_multi(self, row_idx: int) -> None:
    for i, block_table in enumerate(self.block_tables):
        block_table.reset_row(row_idx)