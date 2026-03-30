import importlib.util
import shutil
import sys
from importlib.abc import Loader, MetaPathFinder
from pathlib import Path
from typing import Optional

_PATCHED = False


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    shutil.copymode(src, dst)


def _mindie_base_dir() -> Optional[Path]:
    spec = importlib.util.find_spec("mindie_llm")
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec.origin:
        return Path(spec.origin).parent
    return None


def apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    base_dir = _mindie_base_dir()
    if base_dir is None:
        print("[UCM][MindIE] mindie_llm not found; skip patching", file=sys.stderr)
        return

    src_root = Path(__file__).resolve().parent.parent
    try:
        _copy_file(
            src_root / "uc_utils.py",
            base_dir / "text_generator" / "mempool" / "uc_utils.py",
        )
        _copy_file(
            src_root / "unifiedcache_mempool.py",
            base_dir / "text_generator" / "mempool" / "unifiedcache_mempool.py",
        )
        _copy_file(
            Path(__file__).resolve().parent / "prefix_cache_plugin.py",
            base_dir
            / "text_generator"
            / "plugins"
            / "prefix_cache"
            / "prefix_cache_plugin.py",
        )
        _PATCHED = True
        print("[UCM][MindIE] patch applied to mindie_llm")
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"[UCM][MindIE] Patch application failed: {exc}", file=sys.stderr)


class MindiePatchLoader(Loader):
    def __init__(self, real_loader):
        self.real_loader = real_loader

    def create_module(self, spec):
        return self.real_loader.create_module(spec)

    def exec_module(self, module):
        self.real_loader.exec_module(module)
        try:
            apply_patch()
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"[UCM][MindIE] Patch loader failed: {exc}", file=sys.stderr)


class MindieImportTrigger(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "mindie_llm":
            return None

        meta_path = sys.meta_path[:]
        try:
            sys.meta_path = [
                x for x in sys.meta_path if not isinstance(x, MindieImportTrigger)
            ]
            real_spec = importlib.util.find_spec(fullname, path)
            if real_spec and real_spec.loader:
                real_spec.loader = MindiePatchLoader(real_spec.loader)
                return real_spec
        finally:
            sys.meta_path = meta_path

        return None


def install_hook():
    if not any(isinstance(x, MindieImportTrigger) for x in sys.meta_path):
        sys.meta_path.insert(0, MindieImportTrigger())
    if "mindie_llm" in sys.modules:
        apply_patch()
