from __future__ import annotations

import random
from typing import List, Set


class TokenizerError(RuntimeError):
    """Unified tokenizer exception."""


class TokenizerBase:
    def count_tokens(self, text: str, include_special: bool = True) -> int: ...
    def get_some_tokens(self, num_tokens: int, seed: int | None = None) -> str: ...


class HuggingFaceTokenizer(TokenizerBase):
    def __init__(self, tokenizer_path: str) -> None:
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerBase

            self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        except Exception as exc:
            raise TokenizerError(
                f"Failed to load tokenizer {tokenizer_path!r}: {exc}"
            ) from exc

        self._safe_token_ids: List[int] | None = None

    def count_tokens(self, text: str, include_special: bool = True) -> int:
        """Count the number of tokens in the text. For performance testing, it's recommended to use include_special=True. (Default)"""
        if not text:
            return 0
        try:
            return len(self._tok.encode(text, add_special_tokens=include_special))
        except Exception as exc:
            raise TokenizerError(f"Token counting failed: {exc}") from exc

    def get_some_tokens(self, num_tokens: int, seed: int | None = None) -> str:
        """
        Generate random token text with specified quantity.
        :param num_tokens: Expected number of tokens to generate.
        :param seed: Random seed. If this value is fixed, the generated text content remains unchanged.
        """
        if num_tokens <= 0:
            return ""

        # Use local random instance to avoid affecting global random state
        rng = random.Random(seed)

        safe_ids = self._get_or_build_safe_ids()

        # 1. Randomly select token IDs(Ensure the List not include special tokens)
        selected = rng.choices(safe_ids, k=num_tokens)

        # 2. Decode to text
        text = self._tok.decode(selected, skip_special_tokens=True)

        # 3. Length calibration: Some tokenizers may merge after decode then encode
        encoded = self._tok.encode(text, add_special_tokens=False)
        if len(encoded) > num_tokens:
            text = self._tok.decode(encoded[:num_tokens], skip_special_tokens=True)

        return text

    def _get_or_build_safe_ids(self) -> List[int]:
        if self._safe_token_ids is None:
            # Sort during construction to ensure seed works consistently across different environments
            ids = self._build_safe_token_ids()
            ids.sort()
            self._safe_token_ids = ids
        return self._safe_token_ids

    def _build_safe_token_ids(self) -> List[int]:
        """Filter non-special and decodable token IDs."""
        vocab = self._tok.get_vocab()
        all_ids: Set[int] = set(vocab.values())
        special_ids: Set[int] = set(self._tok.all_special_ids)

        safe_ids_set = all_ids - special_ids

        safe_ids: List[int] = []
        for tid in safe_ids_set:
            try:
                # Filter out control characters or empty characters that cannot be displayed properly
                decoded = self._tok.decode([tid], skip_special_tokens=True).strip()
                if decoded:
                    safe_ids.append(tid)
            except:
                continue

        return safe_ids or list(safe_ids_set)


if __name__ == "__main__":
    model_path = "D:/Models/Qwen3-32B"
    tok_tool = HuggingFaceTokenizer(model_path)

    test_text = "Hello, I am a tester."

    # 1. Count tokens (including special tokens)
    count_with_special = tok_tool.count_tokens(test_text, include_special=True)
    # 2. Count tokens (excluding special tokens)
    count_without_special = tok_tool.count_tokens(test_text, include_special=False)

    print(f"Original text: {repr(test_text)}")
    print(f"Token count (with BOS/EOS): {count_with_special}")
    print(f"Token count (plain text): {count_without_special}")

    # 3. Generate 100 random tokens for stress testing
    random_input = tok_tool.get_some_tokens(100)
    actual_len = tok_tool.count_tokens(random_input, include_special=False)
    print(f"\nGenerated random text length: {actual_len}")
    print(f"Random text snippet: {repr(random_input[:50])}...")

    print("--- Seed Test ---")
    seed_val = 42
    text_a = tok_tool.get_some_tokens(10, seed=seed_val)
    text_b = tok_tool.get_some_tokens(10, seed=seed_val)
    text_c = tok_tool.get_some_tokens(10, seed=114514)

    print(f"Seed({seed_val}) A: {repr(text_a)}")
    print(f"Seed({seed_val}) B: {repr(text_b)}")
    print(f"Seed(123)C: {repr(text_c)}")
