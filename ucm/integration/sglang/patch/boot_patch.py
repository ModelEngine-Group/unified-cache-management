"""Install the SGLang import trigger; patch application remains opt-in."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys


class _SGLangPatchLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        from ucm.integration.sglang.patch.apply_patch import apply_all_patches

        apply_all_patches()


class _SGLangImportTrigger(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang":
            return None
        saved = sys.meta_path[:]
        sys.meta_path = [finder for finder in sys.meta_path if finder is not self]
        try:
            spec = importlib.util.find_spec(fullname, path)
        finally:
            sys.meta_path = saved
        if spec is not None and spec.loader is not None:
            spec.loader = _SGLangPatchLoader(spec.loader)
        return spec


def install_hook() -> None:
    if not any(isinstance(item, _SGLangImportTrigger) for item in sys.meta_path):
        sys.meta_path.insert(0, _SGLangImportTrigger())
