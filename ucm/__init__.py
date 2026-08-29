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

import re

from ucm.integration.vllm.patch.logger_patch import patch_logger as patch_logger

_UCM_BACKEND_DIST = re.compile(
    r"uc-manager-(?:cuda(?:-[a-z0-9]+)*|cann(?:[0-9]+)?-a[0-9]+(?:-[a-z0-9]+)*)"
)


def _is_backend_distribution(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return _UCM_BACKEND_DIST.fullmatch(normalized) is not None


def _guard_single_backend() -> None:
    """Prevent silent file overwrite when multiple backend dists co-exist."""
    from importlib.metadata import distributions

    found = sorted(
        {
            name
            for dist in distributions()
            if (name := (dist.metadata["Name"] or "").lower().replace("_", "-"))
            and _is_backend_distribution(name)
        }
    )
    if len(found) > 1:
        raise ImportError(
            f"Multiple UCM backend distributions are installed: {', '.join(found)}.\n"
            f"They provide the same top-level 'ucm' package and will overwrite each "
            f"other's files. Create a new virtual environment and install exactly "
            f"one uc-manager extra."
        )


_guard_single_backend()
