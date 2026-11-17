"""Add ``req_sparsed_slots`` / ``use_pmr`` fields to vLLM 0.23.0 ``SchedulerOutput``.

Replicates the ``vllm/v1/core/sched/output.py`` hunk of
``vllm-adapt-1020-c8.patch`` at runtime so no ``git apply`` is needed.

``SchedulerOutput`` is a ``@dataclass``; the GSA scheduler constructs it with
two extra keyword arguments. Instead of rebuilding the dataclass-generated
``__init__`` we wrap it to pop the new kwargs and set them as attributes,
preserving the original signature for everything else.
"""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _make_patched_init(orig_init):
    def patched_init(self, *args, **kwargs):
        req_sparsed_slots = kwargs.pop("req_sparsed_slots", None)
        use_pmr = kwargs.pop("use_pmr", False)
        orig_init(self, *args, **kwargs)
        self.req_sparsed_slots = req_sparsed_slots
        self.use_pmr = use_pmr

    return patched_init


@when_imported("vllm.v1.core.sched.output")
def _patch_sched_output(mod):
    if not hasattr(mod, "SchedulerOutput"):
        logger.warning("Skip SchedulerOutput GSA fields patch: SchedulerOutput missing")
        return
    patch_or_inject(
        mod.SchedulerOutput,
        "__init__",
        _make_patched_init(mod.SchedulerOutput.__init__),
    )
    logger.info(
        "UCM GSA SchedulerOutput req_sparsed_slots/use_pmr fields patch applied"
    )
