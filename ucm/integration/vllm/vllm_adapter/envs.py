import os
from typing import Any, Callable

environment_variables = dict[str, Callable[[], Any]] = {
    # Controls whether or not refined profile run
    # are used for preventing OOM
    "VLLM_USE_REFINED_PROFILE": lambda: bool(
        int(os.getenv("VLLM_USE_REFINED_PROFILE", "0"))
    )
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def set_vllm_use_refined_profile(use_refined_profile: bool):
    os.environ["VLLM_USE_REFINED_PROFILE"] = "1" if use_refined_profile else "0"
