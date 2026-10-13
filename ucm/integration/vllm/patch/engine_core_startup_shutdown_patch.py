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
"""Reap workers when EngineCore initialization fails on vLLM >= 0.24.

``EngineCore.__init__`` creates ``self.model_executor`` (which spawns worker
processes) before initializing KV caches and the scheduler.  A KV connector
that raises during ``Scheduler(...)`` therefore leaves live workers behind
while the ``engine_core = EngineCoreProc(...)`` assignment never completes, so
``run_engine_core`` sees ``engine_core is None`` and its ``finally`` block
skips ``engine_core.shutdown()``.

Worker reaping is then left to garbage collection of the half-built engine.
From 0.24 onwards that path can block indefinitely: the parent waits in
``waitpid`` while the workers stay in ``ppoll``, so the process never exits and
device memory is never released.

This patch closes the gap by shutting the executor down explicitly before the
original exception propagates.

It also runs the shared-buffer resource precheck at the very start of
``EngineCore.__init__`` (before the executor spawns workers and before any
model is loaded): vLLM only constructs KV connectors during
``initialize_from_config``, i.e. after ``load_model``, so without this hook a
misconfigured shared buffer surfaces only minutes into startup, after the
whole model has been loaded.
"""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _extract_vllm_config(*args, **kwargs):
    """Locate the VllmConfig argument across EngineCore.__init__ signatures."""
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is not None:
        return vllm_config
    for arg in args:
        if arg is not None and hasattr(arg, "kv_transfer_config"):
            return arg
    return None


def _run_shared_buffer_precheck(*args, **kwargs) -> None:
    """Fail shared-buffer misconfigurations before workers or model load."""
    vllm_config = _extract_vllm_config(*args, **kwargs)
    if vllm_config is None:
        return
    try:
        # Deferred import: this module loads while vllm is still importing
        # (boot_patch), and ucm_connector pulls vllm and torch.
        from ucm.integration.vllm.ucm_connector import (
            precheck_shared_buffer_resources,
        )

        precheck_shared_buffer_resources(vllm_config)
    except RuntimeError:
        # Actionable resource misconfiguration: abort the launch here,
        # before the executor starts and any model is loaded.
        raise
    except Exception:
        # The precheck must never break unrelated startups; the
        # connector-side check remains the authority.
        logger.debug("Skip launch-time shared-buffer precheck.", exc_info=True)


@when_imported("vllm.v1.engine.core")
def patch_engine_core_init(mod):
    engine_core_cls = getattr(mod, "EngineCore", None)
    if engine_core_cls is None:
        return
    original = getattr(engine_core_cls, "__init__", None)
    if original is None:
        return

    def wrapped_init(self, *args, **kwargs):
        _run_shared_buffer_precheck(*args, **kwargs)
        try:
            original(self, *args, **kwargs)
        except BaseException:
            executor = getattr(self, "model_executor", None)
            shutdown = getattr(executor, "shutdown", None) if executor else None
            if callable(shutdown):
                logger.error(
                    "EngineCore init failed after the executor started; shutting "
                    "down workers so the engine process can exit."
                )
                try:
                    shutdown()
                except Exception:
                    logger.exception(
                        "Executor shutdown failed while handling an EngineCore "
                        "init failure."
                    )
            raise

    patch_or_inject(engine_core_cls, "__init__", wrapped_init)
    logger.info(
        "UCM patched EngineCore.__init__ for startup-failure worker shutdown "
        "and launch-time shared-buffer precheck"
    )
