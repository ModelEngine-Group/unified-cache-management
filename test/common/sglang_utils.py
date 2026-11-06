"""Shared source loading and fake modules for SGLang tests."""

import __future__

import ast
import copy
import hashlib
import json
import logging
import re
import sys
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
        dataclass=dataclass,
        replace=replace,
        field=field,
        np=np,
        copy=copy,
        hashlib=hashlib,
        json=json,
        logging=logging,
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
adapter_module = load_source(
    "ucm/integration/sglang/ucm_direct_store_adapter.py",
    AsyncIoObservation=metrics_module.AsyncIoObservation,
    record_metrics=metrics_module.record_metrics,
)
linker_module = load_source(
    "ucm/integration/sglang/ucm_direct_linker.py",
    LinkerOperation=metrics_module.LinkerOperation,
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
