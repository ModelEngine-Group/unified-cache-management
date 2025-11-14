import os
import threading
from typing import Any, Dict

import yaml


class ConfigUtils:
    """
    Singleton Configuration Utility
    Provides methods to read and access YAML configuration files.
    """

    _instance = None
    _lock = threading.Lock()  # Ensure thread-safe singleton creation

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

    def _load_config(self) -> Dict[str, Any]:
        """Internal method to read configuration from file"""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
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
