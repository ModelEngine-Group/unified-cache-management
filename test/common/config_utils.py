import os
import re
import threading
from typing import Any, Dict, Union

import yaml


def _parse_string_type(value: str) -> Any:
    """Convert string values to specific types (bool, int, float) if applicable."""
    lower_val = value.lower().strip()

    # Boolean conversion
    if lower_val in ("true", "yes", "on"):
        return True
    if lower_val in ("false", "no", "off"):
        return False

    # Numeric conversion
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


class ConfigUtils:
    """
    Singleton Configuration Utility
    Provides methods to read and access YAML configuration files.
    Support environment variable ${VAR_NAME:-default_value}
    """

    _instance = None
    _lock = threading.Lock()  # Ensure thread-safe singleton creation
    ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")

    def __init__(self):
        self._config = None

    def __new__(cls, config_file: str = None):
        # Double-checked locking
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init_config(config_file)
                    cls._instance = instance
        return cls._instance

    def _init_config(self, config_file: str = None):
        """Initialize configuration file path and load config"""
        if config_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "..", "config.yaml")

        self.config_file = os.path.abspath(config_file)
        self._config = None  # Lazy load

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively traverse the config to find and replace environment variables."""
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(i) for i in data]
        elif isinstance(data, str):
            return self._replace_and_convert(data)
        return data

    def _replace_and_convert(self, value: str) -> Any:
        """Perform regex substitution and then attempt type conversion."""
        match = self.ENV_PATTERN.fullmatch(value)

        # Case 1: The entire string is a single variable ${VAR:-default}
        if match:
            env_var = match.group(1)
            default_val = match.group(2)
            res = os.getenv(env_var, default_val)
            if res is None:
                return value  # Return original placeholder if no match/default
            return _parse_string_type(res)
        else:
            return None

    def _load_config(self) -> Dict[str, Any]:
        """Internal method to read configuration from file"""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
                return self._substitute_env_vars(raw_config)
        except FileNotFoundError:
            print(f"[WARN] Config file not found: {self.config_file}")
            return {}
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML config: {e}")
            return {}

    def read_config(self) -> Dict[str, Any]:
        """Read configuration file (lazy load)"""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def reload_config(self):
        """Force reload configuration file"""
        self._config = self._load_config()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get top-level configuration item"""
        config = self.read_config()
        return config.get(key, default)

    def get_nested_config(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration, e.g., 'influxdb.host'"""
        config = self.read_config()
        keys = key_path.split(".")
        value = config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


# Global instance
config_utils = ConfigUtils()

if __name__ == "__main__":
    print("DataBase config:", config_utils.get_config("database"))
    print(
        "DataBase host:", config_utils.get_nested_config("database.host", "localhost")
    )
    llm_conn = config_utils.get_config("llm_connection")

    print(f"Old Model Name: {llm_conn['model']}")
    os.environ["SERVER_MODEL_NAME"] = "DeepSeek V3"
    os.environ["LLM_EX_INFO"] = "PC-GSA"
    os.environ["LLM_IGNORE_EOS"] = "TRue"
    print(f"New Model Name-1: {config_utils.get_config('llm_connection')['model']}")
    config_utils.reload_config()
    print(f"New Model Name-2: {config_utils.get_config('llm_connection')['model']}")
    print(f"EX INFO:{config_utils.get_nested_config('llm_connection.ex_info')}")
    print(f"IGNORE_EOS:{config_utils.get_nested_config('llm_connection.ignore_eos')}")
