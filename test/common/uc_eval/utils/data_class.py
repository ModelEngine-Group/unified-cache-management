from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DatasetType(str, Enum):
    """
    The dataset type of uc_eval, including synthetic, multi-turn dialogue, and document-QA.
    """

    SYNTHETIC = "synthetic"
    MULTI_DIALOGUE = "multi_turn_dialogue"
    DOC_QA = "doc_qa"


class BenchmarkModeType(str, Enum):
    """
    The benchmark mode of uc_eval, including evaluate, stable-perf, and default-perf.
    """

    EVAL = "evaluate"
    STABLE_PREF = "stable-perf"
    DEFAULT_PERF = "default-perf"


@dataclass
class ModelConfig:
    ip_ports: str = ""
    tokenizer_path: str = ""
    served_model_name: str = ""
    enable_clear_hbm: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    data_type: str = ""
    dataset_file_path: str = ""
    enable_prefix_cache: str = False
    parallel_num: int = 1
    benchmark_mode: str = "evaluate"
    metrics: Optional[List[str]] = field(default_factory=list)
    eval_class: Optional[str] = None


@dataclass
class PerfConfig:
    data_type: str = ""
    dataset_file_path: str = ""
    enable_prefix_cache: bool = False
    parallel_num: int | List[int] = 1
    prompt_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    prefix_cache_num: List[float] = field(default_factory=list)
    benchmark_mode: str = ""


@dataclass
class SynthericParams:
    """
    The parameters for synthetic dataset
    """

    parallel_num: int = -1
    # The number of tokens for total prompts
    prompt_tokens: int = -1
    # The number of tokens for prefix cache
    prefix_cache_tokens: int = -1
    # List of seeds, to ensure the prefix cache is consistent between warmup and inference
    seeds: list[int] = field(default_factory=list)

    def to_dict(self):
        return vars(self)


@dataclass
class RequestRecord:
    """
    The record for single request
    """

    case_name: str = ""
    request_id: str = ""
    input_data: Optional[str] = ""
    input_tokens: int = 0
    # The real output
    output_data: str = ""
    output_tokens: int = 0
    # The expected output
    expected_output: str = ""
    # The question of the request
    question: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    # The cost of the request
    req_cost: float = 0.0
    # Time to first token, cost of the prefill
    prefill_latency: float = 0.0
    # Time between tokens
    tbt_list: list[float] = field(default_factory=list)
    # Average latency of the tbt_list
    tbt_latency: float = 0.0
    # Whether the request is successful
    is_success: bool = False

    def to_dict(self):
        return vars(self)


@dataclass
class MultiTurnDialogRecord(RequestRecord):
    """
    The record for multi-turn dialogue request
    """

    # The total turn of the conversation
    total_turns: int = -1
    # The current turn of the dialog
    turn_id: int = -1
    # The input content of this dialog, which deletes the history information
    in_content: str = ""
    # If this request belongs to QA dialog
    is_qa: bool = False

    def to_dict(self):
        return vars(self)

@dataclass
class LatencyStatistics:
    """
    the latency statistics of all requests
    """

    # The total latency of all requests(ms)
    e2e_latency_all: float = -1
    # The end to end average throughput(tokens/s)
    output_token_throughput: float = -1
    # The average throughput of all requests(tokens/s)
    token_throughput_per_request: float = -1
    # The TP50 latency of time to first tokens(ms)
    p50_prefill_latency: float = -1
    # The TP90 latency of time to first tokens(ms)
    p90_prefill_latency: float = -1
    # The TP99 latency of time to first tokens(ms)
    p99_prefill_latency: float = -1
    # The max latency of time to first tokens(ms)
    max_prefill_latency: float = -1
    # The average latency of time to first tokens(ms)
    avg_prefill_latency: float = -1
    # The TP50 latency of decoder latency(ms)
    p50_decode_latency: float = -1
    # The TP90 latency of decoder latency(ms)
    p90_decode_latency: float = -1
    # The TP99 latency of decoder latency(ms)
    p99_decode_latency: float = -1
    # The max latency of decoder latency(ms)
    max_decode_latency: float = -1
    # The average latency of decoder latency(ms)
    avg_decode_latency: float = -1

    def to_dict(self):
        return vars(self)
