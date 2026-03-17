import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder


class VllmPatchLoader(Loader):
    def __init__(self, real_loader):
        self.real_loader = real_loader

    def create_module(self, spec):
        return self.real_loader.create_module(spec)

    def exec_module(self, module):
        self.real_loader.exec_module(module)

        try:
            from ucm.integration.vllm.patch.apply_patch import apply_all_patches

            apply_all_patches()
        except Exception as e:
            raise


class VllmImportTrigger(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "vllm":
            return None

        meta_path = sys.meta_path[:]
        try:
            sys.meta_path = [
                x for x in sys.meta_path if not isinstance(x, VllmImportTrigger)
            ]

            real_spec = importlib.util.find_spec(fullname, path)

            if real_spec and real_spec.loader:
                real_spec.loader = VllmPatchLoader(real_spec.loader)
                return real_spec
        finally:
            sys.meta_path = meta_path

        return None


def install_hook():
    if not any(isinstance(x, VllmImportTrigger) for x in sys.meta_path):
        sys.meta_path.insert(0, VllmImportTrigger())
