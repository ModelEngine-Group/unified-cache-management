import math

import torch
from vllm import envs
from vllm.forward_context import get_forward_context


def process_qkv(
    self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:

    if envs.VLLM_USE_REROPE:
        attn_metadata = get_forward_context().attn_metadata
        REROPE_WINDOW = envs.REROPE_WINDOW
        TRAINING_LENGTH = envs.TRAINING_LENGTH
        if attn_metadata and next(iter(attn_metadata.values())).use_rerope:
            q *= (
                ((positions + 1)[:, None].log() / math.log(TRAINING_LENGTH))
                .clip(1)
                .to(q.dtype)
            )
            q2 = q.clone()
            k2 = k.clone()
            k0 = k.clone()

            q, k = self.rotary_emb(positions, q, k)
            q2, _ = self.rotary_emb(positions * 0 + REROPE_WINDOW, q2, k2)
            del k2
        else:
            k0 = k
            q, k = self.rotary_emb(positions, q, k)
            q2 = q

        attn_output = self.attn(q, k, v, query2=q2, key2=k0)
    else:
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

    return attn_output
