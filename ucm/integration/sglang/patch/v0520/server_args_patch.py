"""Expose the dynamic linker selector and JSON config on patched SGLang releases."""

from __future__ import annotations

import json
from copy import deepcopy

from .linker_config import UnifiedCacheLinkerConfig


def apply(*, allow_legacy_ucm: bool = False) -> None:
    import sglang.srt.server_args as server_args

    original_add = server_args.add_cli_args_from_dataclass
    if getattr(original_add, "_ucm_patched", False):
        return

    def add_cli_args_from_dataclass(parser, cls, *args, **kwargs):
        original_add(parser, cls, *args, **kwargs)
        if cls is not server_args.ServerArgs:
            return
        backend = next(
            (
                action
                for action in parser._actions
                if action.dest == "unified_cache_external_linker_backend"
            ),
            None,
        )
        if backend is None:
            raise RuntimeError("SGLang linker backend CLI action is missing.")
        extra_choices = ["dynamic", "ucm"] if allow_legacy_ucm else ["dynamic"]
        backend.choices = list(
            dict.fromkeys([*(backend.choices or ()), *extra_choices])
        )
        parser.add_argument(
            "--unified-cache-external-linker-config",
            type=json.loads,
            default=None,
            help=(
                "JSON object for the dynamic linker: linker, "
                "linker_module_path, and optional linker_extra_config."
            ),
        )

    add_cli_args_from_dataclass._ucm_patched = True
    server_args.add_cli_args_from_dataclass = add_cli_args_from_dataclass

    original_from_cli = server_args.ServerArgs.from_cli_args.__func__

    def from_cli_args(cls, args):
        config = getattr(args, "unified_cache_external_linker_config", None)
        if config is not None and not isinstance(config, dict):
            raise ValueError(
                "--unified-cache-external-linker-config must be a JSON object."
            )
        if args.unified_cache_external_linker_backend == "dynamic" and (
            args.enable_unified_cache_external_linker or config is not None
        ):
            UnifiedCacheLinkerConfig.from_dict(config)
        elif config is not None and args.enable_unified_cache_external_linker:
            raise ValueError(
                "--unified-cache-external-linker-config requires the dynamic "
                "backend when the UCM SGLang patch is enabled."
            )
        result = original_from_cli(cls, args)
        if config is not None:
            result.__dict__["_ucm_linker_config"] = deepcopy(config)
        return result

    server_args.ServerArgs.from_cli_args = classmethod(from_cli_args)
