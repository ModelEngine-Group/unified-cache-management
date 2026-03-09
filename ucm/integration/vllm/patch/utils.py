import importlib.abc
import importlib.util
import sys
from collections import defaultdict

import wrapt

from vllm.logger import init_logger

logger = init_logger(__name__)

_POST_IMPORT_HOOKS = defaultdict(list)


class HookingFinder(importlib.abc.MetaPathFinder):
    """
    A custom Finder that intercepts the import process to wrap the Loader.
    It doesn't load code itself but attaches a post-load hook to the module's loader.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname not in _POST_IMPORT_HOOKS:
            return None

        meta_path_original = sys.meta_path[:]
        sys.meta_path = [x for x in sys.meta_path if not isinstance(x, HookingFinder)]

        try:
            spec = importlib.util.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None

            orig_loader = spec.loader

            class PostImportLoader(importlib.abc.Loader):
                """Wraps the original loader to execute hooks after exec_module."""

                def create_module(self, spec):
                    return orig_loader.create_module(spec)

                def exec_module(self, module):
                    orig_loader.exec_module(module)

                    if fullname in _POST_IMPORT_HOOKS:
                        try:
                            for hook in _POST_IMPORT_HOOKS[fullname]:
                                hook(module)
                        except Exception as e:
                            logger.error(
                                f"Error executing post-import hooks for {fullname}: {e}"
                            )
                            raise

            spec.loader = PostImportLoader()
            return spec

        finally:
            sys.meta_path = meta_path_original
            sys.path_importer_cache.clear()


class PatchOpProxy(wrapt.ObjectProxy):
    """
    Specifically designed for patching PyTorch operators (torch.ops).
    This is tailored for operators registered via vLLM's 'direct_register_custom_op'.
    """

    def __init__(self, wrapped, impl, fake_impl=None):
        super(PatchOpProxy, self).__init__(wrapped)
        self._self_impl = impl
        self._self_fake_impl = fake_impl or getattr(wrapped, "fake_impl", None)

    def __call__(self, *args, **kwargs):
        return self._self_impl(*args, **kwargs)

    @property
    def fake_impl(self):
        return self._self_fake_impl


_FINDER = HookingFinder()
if not any(isinstance(x, HookingFinder) for x in sys.meta_path):
    sys.meta_path.insert(0, _FINDER)


def when_imported(module_name):
    """
    Decorator to register a function to be called as soon as a module is imported.
    """

    def decorator(func):
        if module_name not in _POST_IMPORT_HOOKS:
            _POST_IMPORT_HOOKS[module_name] = []
        _POST_IMPORT_HOOKS[module_name].append(func)

        if module_name in sys.modules:
            mod = sys.modules[module_name]
            if not getattr(mod, "_ucm_patched", False):
                setattr(mod, "_ucm_patched", True)
                func(mod)
        return func

    return decorator


def patch_dataclass_fields(
    target_cls, src_cls, *, include_methods=True, include_ext=True
):
    """
    Structural migration for dataclasses.
    Designed to replace or extend dataclass field definitions from src_cls to target_cls.
    """

    target_cls.__annotations__ = getattr(src_cls, "__annotations__", {})
    target_cls.__dataclass_fields__ = getattr(src_cls, "__dataclass_fields__", {})
    target_cls.__dataclass_params__ = getattr(src_cls, "__dataclass_params__", None)

    for method_name in ["__init__", "__post_init__"]:
        if hasattr(src_cls, method_name):
            setattr(target_cls, method_name, getattr(src_cls, method_name))

    if include_methods:
        for method_name in [
            "__repr__",
            "__eq__",
            "__hash__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        ]:
            if hasattr(src_cls, method_name):
                setattr(target_cls, method_name, getattr(src_cls, method_name))

    if include_ext and hasattr(src_cls, "__match_args__"):
        target_cls.__match_args__ = src_cls.__match_args__

    return target_cls


def _get_replace_wrapper(new_func):
    def wrapper(wrapped, instance, args, kwargs):
        if instance is not None:
            return new_func(instance, *args, **kwargs)
        return new_func(*args, **kwargs)

    return wrapper


def patch_or_inject(target_obj, func_name, replacement_func):
    """
    A general-purpose utility to modify objects or modules.
    If the function exists, it wraps it;
    if it doesn't exist, it injects it.
    """
    if hasattr(target_obj, func_name):
        wrapt.wrap_function_wrapper(
            target_obj, func_name, _get_replace_wrapper(replacement_func)
        )
        logger.debug(
            f"Wrapped: {getattr(target_obj, '__name__', 'module')}.{func_name}"
        )
    else:
        setattr(target_obj, func_name, replacement_func)
        logger.debug(
            f"Injected: {getattr(target_obj, '__name__', 'module')}.{func_name}"
        )
