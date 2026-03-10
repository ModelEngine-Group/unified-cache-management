from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_from_preemption: list[bool]
    new_token_ids: list[list[int]]
    new_block_ids: list[Optional[tuple[list[int], ...]]]
    num_computed_tokens: list[int]
    num_output_tokens: list[int] = field(default_factory=list)

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )
