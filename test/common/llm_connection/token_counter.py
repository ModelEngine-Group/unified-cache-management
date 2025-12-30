# token_counter.py
from __future__ import annotations

import random
from typing import List, Sequence


class TokenizerError(RuntimeError):
    """Unified exception for any tokenization failure."""


class TokenizerBase:
    def count_tokens(self, text: str) -> int: ...
    def get_some_tokens(self, num_tokens: int) -> str: ...


class HuggingFaceTokenizer(TokenizerBase):
    def __init__(self, tokenizer_path: str) -> None:
        try:
            # Load it when needed, so that other forms of Tokenizer can use this dependency in the future
            from transformers import AutoTokenizer, PreTrainedTokenizerBase

            self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=False
            )
        except Exception as exc:
            raise TokenizerError(
                f"Load tokenizer {tokenizer_path!r} failed: {exc}"
            ) from exc
        self._safe_token_ids: List[int] | None = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._tok.encode(text, add_special_tokens=False))
        except Exception as exc:
            raise TokenizerError("Token count failed") from exc

    def get_some_tokens(self, num_tokens: int) -> str:
        if num_tokens <= 0:
            return ""
        safe_ids = self._get_or_build_safe_ids()
        selected = random.choices(safe_ids, k=num_tokens)
        text = self._tok.decode(selected, skip_special_tokens=False)
        truncated = self._tok.encode(text, add_special_tokens=False)[:num_tokens]
        return self._tok.decode(truncated, skip_special_tokens=False)

    def _get_or_build_safe_ids(self) -> List[int]:
        if self._safe_token_ids is None:
            self._safe_token_ids = self._build_safe_token_ids()
        return self._safe_token_ids

    def _build_safe_token_ids(self, sample_size: int = 10_000) -> List[int]:
        vocab_ids: List[int] = list(self._tok.get_vocab().values())
        candidates = random.sample(vocab_ids, min(sample_size, len(vocab_ids)))
        safe: List[int] = []
        for tid in candidates:
            try:
                decoded = self._tok.decode(
                    [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if self._tok.encode(decoded, add_special_tokens=False) == [tid]:
                    safe.append(tid)
            except Exception:
                continue
        if not safe:
            special_ids = set(getattr(self._tok, "all_special_ids", []))
            safe = [i for i in vocab_ids if i not in special_ids] or vocab_ids
        return safe


if __name__ == "__main__":
    tok = HuggingFaceTokenizer("D:/Models/Qwen3-32B")
    text = "Hello, how are you?"
    print("Text:", repr(text))
    print("Tokens:", tok.count_tokens(text))

    random_text = tok.get_some_tokens(4096)
    print("Random 10 tokens:", repr(random_text))
    print("Actual tokens:", tok.count_tokens(random_text))
