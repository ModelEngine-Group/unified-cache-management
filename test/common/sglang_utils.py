"""Shared source loading and fake modules for SGLang tests."""

import __future__

import ast
import copy
import hashlib
import importlib
import json
import logging
import re
import sys
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, List

import numpy as np
from common.path_utils import get_test_root


def _load_source(relative_path, definitions=None, **dependencies):
    """Load selected Python definitions with caller-provided dependencies."""
    path = get_test_root().parent / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if definitions is not None:
        tree.body = [
            node for node in tree.body if getattr(node, "name", None) in definitions
        ]
    module = types.ModuleType("_test_" + path.stem)
    module.__dict__.update(dependencies)
    sys.modules[module.__name__] = module
    exec(
        compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag),
        module.__dict__,
    )
    return module


def load_source(relative_path, definitions=None, **dependencies):
    defaults = dict(
        wraps=wraps,
        dataclass=dataclass,
        replace=replace,
        field=field,
        np=np,
        copy=copy,
        hashlib=hashlib,
        json=json,
        logging=logging,
        init_logger=logging.getLogger,
        time=time,
        re=re,
        threading=threading,
        Path=Path,
        UnifiedCacheLinker=object,
        get_device_module=lambda: None,
        Any=Any,
        List=List,
        Task=object,
        UcmKVStoreBaseV1=object,
        torch=types.SimpleNamespace(Tensor=object),
    )
    defaults.update(dependencies)
    return _load_source(relative_path, definitions=definitions, **defaults)


metrics_module = load_source(
    "ucm/integration/sglang/metrics.py", contextmanager=contextmanager, time=time
)
counter_metrics_module = load_source(
    "ucm/integration/sglang/patch/layerwise_counter_metrics.py",
    importlib=importlib,
    PackageNotFoundError=PackageNotFoundError,
    version=version,
    counter_metrics_enabled=metrics_module.counter_metrics_enabled,
    start_timing=metrics_module.start_timing,
    finish_timing=metrics_module.finish_timing,
)
adapter_module = load_source(
    "ucm/integration/sglang/ucm_direct_store_adapter.py",
    _record_interface_duration=metrics_module._record_interface_duration,
    record_metrics=metrics_module.record_metrics,
)
linker_module = load_source(
    "ucm/integration/sglang/ucm_direct_linker.py",
    LinkerOperation=metrics_module.LinkerOperation,
    _record_interface_duration=metrics_module._record_interface_duration,
    measure_phase=metrics_module.measure_phase,
    start_timing=metrics_module.start_timing,
    finish_timing=metrics_module.finish_timing,
    record_metrics=metrics_module.record_metrics,
    shutdown_metrics_logger=metrics_module.shutdown_metrics_logger,
    create_metrics_logger=metrics_module.create_metrics_logger,
    **{
        name: getattr(adapter_module, name)
        for name in ("UcmDirectStoreAdapter", "UcmLayerTransfer", "UcmPoolMetadata")
    },
)
pipeline_module = load_source(
    "ucm/store/pipeline/connector.py",
    definitions={
        "UcmPipelineStoreTransTask",
        "UcmPipelineStore",
    },
)
