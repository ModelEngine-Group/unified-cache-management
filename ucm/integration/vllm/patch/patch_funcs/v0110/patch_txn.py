#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Best-effort transactional helpers for monkey patching."""

from __future__ import annotations

import types
from collections.abc import Callable, MutableMapping
from typing import Any


class PatchTxn:
    """Track monkey-patch mutations and support full rollback."""

    _MISSING = object()
    _global_undo_stack: list[Callable[[], None]] = []
    _global_seen_keys: set[tuple[int, str]] = set()

    def __init__(self) -> None:
        self._undo_stack: list[Callable[[], None]] = []

    @classmethod
    def _record_global_once(
        cls, key: tuple[int, str], undo: Callable[[], None]
    ) -> None:
        if key in cls._global_seen_keys:
            return
        cls._global_seen_keys.add(key)
        cls._global_undo_stack.append(undo)

    def set_attr(self, obj: Any, name: str, value: Any) -> None:
        old_value = getattr(obj, name, self._MISSING)
        key = (id(obj), f"attr:{name}")

        def _undo() -> None:
            if old_value is self._MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old_value)

        self._record_global_once(key, _undo)
        self._undo_stack.append(_undo)
        setattr(obj, name, value)

    def set_dict_item(
        self, mapping: MutableMapping[Any, Any], key: Any, value: Any
    ) -> None:
        had_key = key in mapping
        old_value = mapping.get(key, self._MISSING)
        dict_key = (id(mapping), f"dict:{repr(key)}")

        def _undo() -> None:
            if had_key:
                mapping[key] = old_value
            else:
                mapping.pop(key, None)

        self._record_global_once(dict_key, _undo)
        self._undo_stack.append(_undo)
        mapping[key] = value

    def swap_function_impl(self, original: Any, replacement: Any) -> bool:
        """Swap a function payload in-place and record rollback."""
        if not isinstance(original, types.FunctionType):
            return False
        if not isinstance(replacement, types.FunctionType):
            return False

        old_code = original.__code__
        old_defaults = original.__defaults__
        old_kwdefaults = original.__kwdefaults__
        old_annotations = dict(getattr(original, "__annotations__", {}))
        old_doc = original.__doc__
        key = (id(original), "function_impl")

        def _undo() -> None:
            original.__code__ = old_code
            original.__defaults__ = old_defaults
            original.__kwdefaults__ = old_kwdefaults
            original.__annotations__ = old_annotations
            original.__doc__ = old_doc

        self._record_global_once(key, _undo)
        self._undo_stack.append(_undo)
        try:
            original.__code__ = replacement.__code__
            original.__defaults__ = replacement.__defaults__
            original.__kwdefaults__ = replacement.__kwdefaults__
            original.__annotations__ = dict(getattr(replacement, "__annotations__", {}))
            original.__doc__ = replacement.__doc__
            return True
        except Exception:
            # Only unwind this single swap attempt.
            self._undo_stack.pop()
            try:
                _undo()
            except Exception:
                pass
            return False

    def rollback(self) -> None:
        while self._undo_stack:
            undo = self._undo_stack.pop()
            try:
                undo()
            except Exception:
                # Best-effort rollback: never mask original apply error.
                pass

    @classmethod
    def rollback_all(cls) -> None:
        while cls._global_undo_stack:
            undo = cls._global_undo_stack.pop()
            try:
                undo()
            except Exception:
                pass
        cls._global_seen_keys.clear()

    def commit(self) -> None:
        self._undo_stack.clear()
