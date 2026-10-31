"""Add GSA/PMR attributes to vLLM 0.23.0 ``Request``.

Replicates the ``vllm/v1/request.py`` hunk of ``vllm-adapt-1020-c8.patch``
at runtime so no ``git apply`` is needed. ``Request`` is a ``@dataclass`` with a
custom ``__init__``; the patch appends a few GSA/PMR instance attributes after
the original initializer runs.
"""

import numpy as np

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _make_patched_init(orig_init):
    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.spec_token_positions: list[int] = []
        self.spec_decode_masks: np.array = []
        self.pmr_tree_search_handle = None
        self.two_grams: dict[int, int] = {}
        self.use_pmr = True

    return patched_init


@when_imported("vllm.v1.request")
def _patch_request(mod):
    if not hasattr(mod, "Request"):
        logger.warning("Skip Request GSA fields patch: Request missing")
        return
    patch_or_inject(mod.Request, "__init__", _make_patched_init(mod.Request.__init__))
    logger.info("UCM GSA Request spec/pmr fields patch applied")
