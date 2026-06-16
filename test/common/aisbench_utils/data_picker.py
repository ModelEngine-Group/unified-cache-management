"""
Data Picker - Randomly select data from GSM8K dataset
Supports non-repeat mode and repeatable mode, with auto-reset feature

Changes: Removed file-based picked_ids.txt dependency, using in-memory tracking only.
When data is exhausted in non-repeat mode, auto-reset is always applied.
"""
import json
import logging
import os
from typing import Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO)


class LightTokenizer:
    """
    Lightweight tokenizer wrapper using the `tokenizers` Rust library.
    Avoids importing `transformers`/`torch`, which causes DLL issues on
    Windows systems without CUDA drivers installed.

    Provides the same public interface as `transformers.AutoTokenizer`
    for the operations used in dataset generation:
      - encode(text, add_special_tokens=False) -> List[int]
      - decode(ids, skip_special_tokens=True) -> str
      - get_vocab() -> Dict[str, int]
      - __len__() -> vocab_size
      - all_special_ids -> List[int]
    """

    def __init__(self, model_path: str) -> None:
        # Resolve tokenizer.json path from model directory
        tokenizer_json = os.path.join(model_path, "tokenizer.json")
        if not os.path.isfile(tokenizer_json):
            raise FileNotFoundError(
                f"tokenizer.json not found in {model_path}"
            )

        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(tokenizer_json)

        # Build special IDs from tokenizer_config.json (if available)
        self._all_special_ids: List[int] = self._load_special_ids(model_path)

    def _load_special_ids(self, model_path: str) -> List[int]:
        """Extract special token IDs from tokenizer_config.json."""
        config_path = os.path.join(model_path, "tokenizer_config.json")
        special_ids: List[int] = []

        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                for sid, info in cfg.get("added_tokens_decoder", {}).items():
                    if info.get("special", False):
                        special_ids.append(int(sid))
            except (json.JSONDecodeError, OSError):
                pass  # non-critical; empty list is safe

        return special_ids

    # ---- Public interface (mirrors transformers.PreTrainedTokenizer) ----

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to a list of token IDs."""
        if not text:
            return []
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text."""
        if not ids:
            return ""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary mapping {token_str: token_id}."""
        return self._tok.get_vocab()

    @property
    def all_special_ids(self) -> List[int]:
        """List of special token IDs."""
        return self._all_special_ids

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self._tok.get_vocab_size()


class DataPicker:
    """
    GSM8K dataset picker

    Features:
    - Non-repeat mode (prefix_flag=1): Pick from unused data randomly, auto-reset when exhausted
    - Repeatable mode (prefix_flag=0): Pick from all data randomly
    - In-memory tracking: No file dependency, picked IDs tracked in memory only
    """

    def __init__(self, jsonl_file: str, prefix_flag: int = 0):
        """
        Initialize the picker

        Args:
            jsonl_file: Path to GSM8K dataset file
            prefix_flag: 1 for non-repeat mode, 0 for repeatable mode
        """
        self.jsonl_file = jsonl_file
        self.prefix_flag = prefix_flag

        self.total_lines = self._count_lines()
        self.picked_ids = set()

    def _count_lines(self) -> int:
        """Count total lines in jsonl file"""
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _save_picked_ids(self):
        """Save picked ID records (in-memory only, no file writing)"""
        # No file operation needed - picked_ids is maintained in memory
        pass

    def get_unpicked_ids(self) -> list:
        """Get all unpicked IDs"""
        all_ids = set(range(self.total_lines))
        return list(all_ids - self.picked_ids)

    def get_usage_ratio(self) -> float:
        """Get usage ratio (picked / total)"""
        if self.total_lines == 0:
            return 0.0
        return len(self.picked_ids) / self.total_lines

    def pick_one(self) -> Optional[str]:
        """
        Randomly pick one data item

        Returns:
            The picked data content, or None if no unpicked data available
        """
        import random

        # Repeatable mode
        if self.prefix_flag == 0:
            selected_id = random.randint(0, self.total_lines - 1)
            return self._read_line(selected_id)

        # Non-repeat mode
        available_ids = self.get_unpicked_ids()

        if not available_ids:
            # Data exhausted - auto-reset
            logging.warning(
                f"Data exhausted ({self.total_lines} items), auto-resetting picked IDs"
            )
            self.picked_ids.clear()
            available_ids = self.get_unpicked_ids()

        # Randomly select an unpicked ID
        selected_id = random.choice(available_ids)

        # Record the picked ID in memory
        self.picked_ids.add(selected_id)

        return self._read_line(selected_id)

    def _read_line(self, line_id: int) -> Optional[str]:
        """Read data from specified line"""
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_id:
                    try:
                        data = json.loads(line)
                        return data.get('question', '')
                    except json.JSONDecodeError:
                        logging.warning(f"Line {line_id} JSON parse failed")
                        return None
        return None

    def reset(self):
        """
        Reset records (clear picked records in memory)
        """
        self.picked_ids.clear()
        logging.info(f"DataPicker reset, all {self.total_lines} items available again")
