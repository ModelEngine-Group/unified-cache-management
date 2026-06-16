"""
AISBench Prefix Cache test configuration data classes
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Union


class DatasetType(str):
    """Dataset type enumeration"""
    NORMAL = "normal"
    PREFIX_CACHE = "prefix_cache"


class TestMode(str):
    """Test mode enumeration"""
    PERFORMANCE = "perf"
    ACCURACY = "accuracy"


@dataclass
class AisbenchConfig:
    """
    AISBench Prefix Cache test configuration

    Contains all test parameters, supports variable-length dataset testing

    Note: Users can pass any reasonable type for numeric/string parameters,
          automatic type conversion will be performed in __post_init__
    """
    # Basic configuration (required)
    input_len: int = 2048              # Input token length
    output_len: int = 2048             # Output token length
    data_num: int = 160                # Number of dataset samples
    concurrency: int = 40              # Maximum concurrency

    # Request configuration
    request_rate: Union[int, float, str] = 0  # Request rate, auto-convert to str
    test_type: str = "stream"          # stream or text
    repeat: int = 1                    # Test repeat count

    # Dataset configuration
    dataset: str = ""                  # Specified dataset path (optional)
    dataset_type: str = DatasetType.NORMAL  # normal or prefix_cache

    # Prefix Cache configuration
    prefix_num: int = 1                # Number of prefix types
    repeat_rate: Union[int, float, str] = 0.5  # Prefix repeat rate, auto-parse (50%, 0.5, 50 all work)
    prefix_test: bool = False          # Whether to warmup prefix first
    dp: int = 1                        # Number of DP domains
    seed: int = 1                      # Random seed

    # Variable-length dataset configuration
    length_mean: Optional[int] = None  # Input length mean (Gaussian distribution)
    length_std: Optional[float] = None # Input length standard deviation
    length_min: Optional[int] = None   # Input length minimum
    length_max: Optional[int] = None   # Input length maximum

    # Test mode configuration
    test_accuracy: bool = False        # Whether to test accuracy
    enable_think: bool = False         # DeepSeek V3.1 thinking mode

    # Hardware configuration
    npu_num: int = 1                   # Number of NPU cards

    # Test name (for result recording)
    test_name: str = "Default"

    def __post_init__(self):
        """Auto-convert types after initialization"""
        # Convert request_rate to string format
        # Accepts: int (0, 10), float (0.5), str ("0", "10", "0.5")
        self._request_rate_raw = self.request_rate  # Store original value
        self.request_rate = self._parse_request_rate(self.request_rate)

        # Convert repeat_rate to string format (percentage or decimal)
        # Accepts: int (50), float (0.5), str ("50%", "0.5", "50")
        self._repeat_rate_raw = self.repeat_rate  # Store original value
        self.repeat_rate = self._parse_repeat_rate(self.repeat_rate)

    def _parse_request_rate(self, value: Union[int, float, str]) -> str:
        """
        Parse request_rate to string format

        Args:
            value: Can be int (0=unlimited, 10=10 req/s), float, or str

        Returns:
            String representation for AISBench
        """
        if isinstance(value, str):
            # Already string, just return it
            return value
        else:
            # Convert int/float to string
            return str(int(value) if isinstance(value, float) and value.is_integer() else value)

    def _parse_repeat_rate(self, value: Union[int, float, str]) -> str:
        """
        Parse repeat_rate to standard format

        Args:
            value: Can be:
                - int: 50 -> "0.5" (interpreted as percentage)
                - float: 0.5 -> "0.5"
                - str: "50%" -> "0.5", "0.5" -> "0.5", "50" -> "0.5"

        Returns:
            Decimal string format (e.g., "0.5", "0.3")
        """
        if isinstance(value, str):
            # Handle string input
            if value.endswith('%'):
                # Percentage format: "50%" -> 0.5
                pct = float(value.rstrip('%'))
                return str(pct / 100)
            else:
                # Already decimal or plain number string
                try:
                    num = float(value)
                    if num > 1:
                        # Assume percentage: "50" -> 0.5
                        return str(num / 100)
                    return str(num)
                except ValueError:
                    return value
        elif isinstance(value, (int, float)):
            # Handle numeric input
            num = float(value)
            if num > 1:
                # Assume percentage: 50 -> 0.5
                return str(num / 100)
            return str(num)
        return str(value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'input_len': self.input_len,
            'output_len': self.output_len,
            'data_num': self.data_num,
            'concurrency': self.concurrency,
            'request_rate': self.request_rate,
            'test_type': self.test_type,
            'repeat': self.repeat,
            'dataset': self.dataset,
            'dataset_type': self.dataset_type,
            'prefix_num': self.prefix_num,
            'repeat_rate': self.repeat_rate,
            'prefix_test': self.prefix_test,
            'dp': self.dp,
            'seed': self.seed,
            'length_mean': self.length_mean,
            'length_std': self.length_std,
            'length_min': self.length_min,
            'length_max': self.length_max,
            'test_accuracy': self.test_accuracy,
            'enable_think': self.enable_think,
            'npu_num': self.npu_num,
            'test_name': self.test_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AisbenchConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AisbenchResult:
    """
    AISBench test result data class
    """
    test_name: str = ""
    status: str = ""

    # Basic parameters
    input_len: int = 0
    output_len: int = 0
    data_num: int = 0
    concurrency: int = 0
    request_rate: str = "0"
    dataset_type: str = ""
    repeat_rate: str = ""

    # Performance metrics
    ttft_avg: float = 99999
    ttft_p90: float = 99999
    tpot_avg: float = 99999
    tpot_p90: float = 99999
    total_time: float = 99999

    # Throughput metrics
    output_throughput: float = 99999
    single_output_throughput: float = 99999
    e2e_throughput: float = 99999
    single_e2e_throughput: float = 99999
    input_token_throughput: float = 99999
    prefill_throughput: float = 99999

    # QPS metrics
    qps: float = 99999
    qpm: float = 99999

    # Token statistics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0

    # Prefix Cache hit rate
    hbm_hit_rate: str = ""
    hbm_hits: int = 0
    hbm_queries: int = 0
    external_hit_rate: str = ""
    external_hits: int = 0
    external_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return vars(self)