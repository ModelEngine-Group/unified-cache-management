import logging
import os
import re
import threading
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def _parse_string_type(value: Any) -> Any:
    """Convert string values to appropriate Python types (bool, int, float)."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return stripped

    lower_val = stripped.lower()

    # Boolean conversion (support multiple formats)
    if lower_val == "true":
        return True
    if lower_val == "false":
        return False

    # Numeric conversion
    try:
        if "." in stripped and stripped.count(".") == 1:
            if not stripped.startswith(".") and not stripped.endswith("."):
                return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


class ConfigUtils:
    """
    Thread-safe singleton configuration utility.

    Features:
        - YAML config with environment variable substitution ${VAR:-default}
        - Automatic type conversion for string values
        - Nested key access using dot notation
    """

    _instance: Optional["ConfigUtils"] = None
    _lock = threading.Lock()
    _init_lock = threading.Lock()

    ENV_PATTERN = re.compile(r"^\$\{(\w+)(?::-([^}]*))?\}$")

    def __new__(cls, config_file: Optional[str] = None, **kwargs):
        """Double-checked locking for thread-safe singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._config = None
                    instance.config_file = None
                    cls._instance = instance
                    instance._init_config(config_file)

        return cls._instance

    def __init__(self):
        self._config = None

    @classmethod
    def get_instance(cls, config_file: Optional[str] = None) -> "ConfigUtils":
        """Get singleton instance"""
        return cls(config_file)

    def _init_config(self, config_file: Optional[str] = None):
        """Initialize config file path"""
        if config_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "..", "config.yaml")

        self.config_file = os.path.abspath(config_file)
        logger.info(f"Configuration file path set to: {self.config_file}")

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables"""
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._process_string_value(data)
        return data

    def _process_string_value(self, value: str) -> Any:
        """Process string: check for env var pattern, then type conversion"""
        match = self.ENV_PATTERN.fullmatch(value.strip())

        if match:
            var_name = match.group(1)
            default_val = match.group(2)

            env_value = os.getenv(var_name)
            if env_value is not None:
                return _parse_string_type(env_value.strip())
            elif default_val is not None:
                return _parse_string_type(default_val.strip())
            else:
                logger.warning(
                    f"Env var '{var_name}' not found, no default. Keeping: {value}"
                )
                return value
        else:
            return _parse_string_type(value)

    def _load_config(self) -> Dict[str, Any]:
        """Load config from file"""
        if not self.config_file:
            logger.error("Config file path not initialized")
            return {}

        if not os.path.exists(self.config_file):
            logger.warning(f"Configuration file not found: {self.config_file}")
            return {}

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
                return self._substitute_env_vars(raw_config)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def read_config(self) -> Dict[str, Any]:
        """Lazy load configuration"""
        if self._config is None:
            with self._init_lock:
                if self._config is None:
                    self._config = self._load_config()
        return self._config

    def reload_config(self) -> Dict[str, Any]:
        """Force reload configuration"""
        with self._init_lock:
            self._config = self._load_config()
            logger.info(f"Configuration reloaded successfully")
            return self._config

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get top-level config item"""
        return self.read_config().get(key, default)

    def get_nested_config(
        self, key_path: str, default: Any = None, separator: str = "."
    ) -> Any:
        """Get nested config using dot notation (e.g., 'database.host')"""
        config = self.read_config()
        keys = key_path.split(separator)
        value = config

        try:
            for key in keys:
                if not isinstance(value, dict):
                    return default
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Support config_utils['key']"""
        return self.get_config(key)

    def __contains__(self, key: str) -> bool:
        """Support 'key' in config_utils"""
        return key in self.read_config()


# Global instance
config_utils = ConfigUtils()

if __name__ == "__main__":
    print("=== Configuration Test ===")
    print(f"Config file path: {config_utils.config_file}")
    print(f"LLM Connection: {config_utils.get_config('llm_connection')}")
    print(f"EXTRA_INFO: {config_utils.get_nested_config('llm_connection.extra_info')}")
    os.environ["LLM_EX_INFO"] = "prefix cache and gsa"
    config_utils.reload_config()
    print(f"EXTRA_INFO: {config_utils.get_nested_config('llm_connection.extra_info')}")
    print(
        f"Timeout: {config_utils.get_nested_config('llm_connection.timeout')} (type: {type(config_utils.get_nested_config('llm_connection.timeout'))})"
    )
