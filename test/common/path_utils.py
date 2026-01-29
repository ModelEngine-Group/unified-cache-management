"""Utility functions for path management in tests."""

import os
from pathlib import Path


def get_test_root() -> Path:
    """Get the /test directory path regardless of where pytest is run from.

    This function locates the test root directory by finding the directory
    containing this module (common/) and moving up one level.

    Returns:
        Path: The absolute path to the /test directory

    Example:
        >>> from common.path_utils import get_test_root
        >>> config_file = get_test_root() / "config.yaml"
        >>> prompt_file = get_test_root() / "suites" / "E2E" / "prompts" / "test.json"
    """
    # Get the directory where this module is located (common/)
    # Then go up one level to reach /test
    return Path(__file__).resolve().parent.parent


def get_path_relative_to_test_root(subdir_path: str | Path) -> Path:
    """Get a path relative to the /test directory.

    Args:
        subdir_path: Relative path from test root (can be string or Path)

    Returns:
        Path: The absolute path to the requested subdirectory/file

    Example:
        >>> from common.path_utils import get_test_subdir
        >>> config_file = get_test_subdir("config.yaml")
        >>> prompt_file = get_test_subdir("suites/E2E/prompts/test.json")
    """
    return get_test_root() / subdir_path


def get_path_to_model(model_name: str, config) -> str:
    return os.path.join("/home/models/", model_name)
