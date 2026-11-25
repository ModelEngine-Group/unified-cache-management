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
"""
Monkey patching module for vLLM to apply UCM patches automatically.
This replaces the need for manual `git apply` commands.
"""

from importlib import import_module
from typing import Optional

from ucm.logger import init_logger

logger = init_logger(__name__)

import os

ENABLE_SPARSE = os.getenv("ENABLE_SPARSE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ENABLE_UCM_PATCH = os.environ.get("ENABLE_UCM_PATCH", "").lower() in ("1", "true")
ENABLE_GSA_PATCH = os.environ.get("ENABLE_GSA_PATCH", "").lower() in ("1", "true")


def _preflight_gsa_cust_opapi() -> bool:
    """预检 libcust_opapi.so 提供 GSA aclnn 符号（部署问题前移到启动期）。"""
    import ctypes

    try:
        # 默认 mode（RTLD_LOCAL）即可：仅检查符号存在性，与 C++ 侧
        # dlopen(RTLD_LAZY) 命中同一进程内已加载的 libcust_opapi.so。
        lib = ctypes.CDLL("libcust_opapi.so")
    except OSError as exc:
        logger.error(
            "GSA custom op lib libcust_opapi.so not loadable (dlopen search = "
            "this process's LD_LIBRARY_PATH). torch.ops._C_gsa."
            "npu_kv_quant_sparse_flash_attention_gsa will fail at the first "
            "decode step with 'aclnnKvQuantSparseFlashAttentionGsa ... not in "
            "libopapi.so'. Fix: (1) bash "
            "ucm/sandbox/sparse/gsav2/ops/ascend/install.sh (installs "
            "GSA-custom_ops*.run into opp/vendors/gsa and writes setenv.bash); "
            "(2) source .../opp/vendors/gsa/setenv.bash, or prepend "
            ".../opp/vendors/gsa/op_api/lib to LD_LIBRARY_PATH, in the service "
            "launch environment (start_online_pdmix.sh / gsa_online.sh / "
            "start_decode_glm52.sh already do this); (3) restart the service. "
            "dlopen error: %s",
            exc,
        )
        return False
    if not hasattr(lib, "aclnnKvQuantSparseFlashAttentionGsaGetWorkspaceSize"):
        logger.error(
            "libcust_opapi.so loaded but lacks aclnnKvQuantSparseFlashAttentionGsa "
            "symbols: the first libcust_opapi.so on this process's "
            "LD_LIBRARY_PATH belongs to another vendor (ucm/custom_transformer/"
            "...). npu_kv_quant_sparse_flash_attention_gsa will fail at the "
            "first decode step. Fix: prepend .../opp/vendors/gsa/op_api/lib to "
            "LD_LIBRARY_PATH (before other vendors) and restart the service."
        )
        return False
    logger.info("GSA custom op lib libcust_opapi.so OK (aclnn symbols present)")
    return True


def _ensure_gsa_ops_loaded() -> None:
    """加载 gsa_ops C++ 扩展，注册 torch.ops._C_gsa 命名空间。"""
    if not ENABLE_GSA_PATCH:
        return
    try:
        import gsa_ops  # noqa: F401  触发 torch.ops._C_gsa 注册

        logger.info("gsa_ops C++ extension loaded; torch.ops._C_gsa registered")
        # 注册成功不等于 aclnn 符号可解析：gsa_ops.so 只在首次调用算子时才
        # dlopen libcust_opapi.so，部署问题会晚到首个 decode step 才爆——
        # 此处预检前移（参照 gsa_on_device import ucm_custom_ops 的显式化模式）。
        _preflight_gsa_cust_opapi()
    except ImportError as e:
        logger.warning(
            "ENABLE_GSA_PATCH=1 but gsa_ops C++ extension not installed; "
            "torch.ops._C_gsa.* ops will be unavailable. "
            "Build & install via `pip install -v -e ucm/sandbox/sparse/gsav2/ops`. "
            "Detail: %s",
            e,
        )


def _strip_build(v: Optional[str]) -> Optional[str]:
    """Strip only build/local metadata (the +xxx suffix), e.g. 0.26.0+empty -> 0.26.0."""
    if not v:
        return None
    return str(v).strip().split("+", 1)[0]


def _norm_version(v: Optional[str]) -> Optional[str]:
    v = _strip_build(v)
    if not v:
        return None
    # common suffixes: 0.11.0.post1 / 0.11.0rc1
    v = v.split(".post", 1)[0]
    v = v.split("rc", 1)[0]
    return v


def _read_vllm_ascend_version_raw() -> Optional[str]:
    """Read vllm_ascend version string, stripping only build metadata (+xxx)."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return _strip_build(version("vllm-ascend"))
        except PackageNotFoundError:
            return None
    except Exception:
        pass

    try:
        import importlib

        mod = importlib.import_module("vllm_ascend")
        return _strip_build(getattr(mod, "__version__", None))
    except Exception:
        return None


def get_vllm_ascend_version_full() -> Optional[str]:
    """Detect vllm_ascend version preserving rc/post suffixes (e.g. 0.18.0rc1)."""
    return _read_vllm_ascend_version_raw()


def get_vllm_ascend_version() -> Optional[str]:
    """Detect normalized vllm_ascend version (e.g. 0.18.0rc1 -> 0.18.0)."""
    return _norm_version(_read_vllm_ascend_version_raw())


_vllm_version: Optional[str] = None


def _read_vllm_version_raw() -> Optional[str]:
    """Read vLLM version string from the installed module, stripping build metadata."""
    try:
        import vllm as vllm_pkg

        return _strip_build(getattr(vllm_pkg, "__version__", None))
    except ImportError:
        logger.warning("vLLM is not installed")
        return None
    except Exception as e:
        logger.warning(f"Failed to detect vLLM version: {e}")
        return None


def get_vllm_version_full() -> Optional[str]:
    """Detect vLLM version preserving rc/post suffixes (e.g. 0.26.0rc1)."""
    return _read_vllm_version_raw()


def get_vllm_version() -> Optional[str]:
    """Detect normalized vLLM version (e.g. 0.26.0+empty / 0.26.0rc1 -> 0.26.0)."""
    global _vllm_version
    if _vllm_version is not None:
        return _vllm_version
    _vllm_version = _norm_version(_read_vllm_version_raw())
    return _vllm_version


def get_supported_versions() -> list[str]:
    """Get patch-required vLLM versions."""
    return [
        "0.11.0",
        "0.17.0",
        "0.18.0",
        "0.19.1",
        "0.20.2",
        "0.21.0",
        "0.22.1",
        "0.23.0",
        "0.24.0",
        "0.25.1",
        "0.26.0",
        "0.27.0",
        "0.27.1",
        "0.28.0",
        "0.30.0",
    ]


def apply_all_patches() -> None:
    """Apply all vLLM patches based on detected version."""
    version: Optional[str] = None
    try:
        from ucm.integration.vllm.patch.logger_patch import patch_logger

        if not ENABLE_UCM_PATCH:
            return

        version = get_vllm_version()
        if version is None:
            raise ValueError("Could not detect vLLM version")

        supported_versions = get_supported_versions()
        if version not in supported_versions:
            logger.warning(
                f"No version-specific vLLM patches available for vLLM {version}. "
                f"Versions applicable for UCM patches: {', '.join(supported_versions)}."
            )

        ascend_version = get_vllm_ascend_version()
        logger.info(
            f"Detected vLLM version: {get_vllm_version_full()} "
            f"(normalized: {version})"
        )
        logger.info(
            f"Detected vllm-ascend version: {get_vllm_ascend_version_full()} "
            f"(normalized: {ascend_version})"
        )
        # vLLM and vllm-ascend share per-version patch directories (v0XYZ), so
        # their versions must align; when they disagree, vLLM is authoritative.
        if ascend_version is not None and ascend_version != version:
            logger.warning(
                f"vllm-ascend version ({ascend_version}) differs from vLLM version "
                f"({version}); aligning vllm-ascend patch selection to {version}."
            )
            ascend_version = version
        # UCM PATCH: vllm-ascend registers UCMConnector as an alias for the
        # concrete UCMConnectorV1 class used by MultiConnector metrics.
        if ascend_version in {
            "0.18.0",
            "0.19.1",
            "0.20.2",
            "0.22.1",
            "0.23.0",
            "0.24.0",
            "0.25.1",
            "0.26.0",
        }:
            logger.info("UCM patching vllm-ascend UCM connector metrics alias...")
            import ucm.integration.vllm.patch.ucm_connector_registration_patch

        # Apply vllm/vllm-ascend version-specific patches
        # vllm patches
        match version:
            case "0.11.0":
                logger.info("UCM patching vllm for pc...")
                import ucm.integration.vllm.patch.v0110.vllm.pc_patch

                if ENABLE_SPARSE:
                    logger.info("UCM patching vllm for sparse...")
                    import ucm.integration.vllm.patch.v0110.vllm.sparse_patch
            case "0.18.0":
                logger.info("UCM patching vllm for pc...")
                import ucm.integration.vllm.patch.v0180.vllm.pc_patch
            case "0.19.1":
                logger.info("UCM patching vllm for pc...")
                import ucm.integration.vllm.patch.v0191.vllm.pc_patch
            case "0.23.0":
                if ENABLE_GSA_PATCH:
                    _ensure_gsa_ops_loaded()
                    logger.info("UCM patching vllm 0.23.0 for GSA sparse attention...")
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.block_table_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.engine_core_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.gpu_model_runner_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.gpu_worker_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.kv_cache_manager_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.mla_attention_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.sched_output_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm.scheduler_patch
                else:
                    logger.info(
                        "UCM GSA vllm patches skipped (ENABLE_GSA_PATCH is not set)."
                    )
            case _:
                pass

        major, minor, *_ = version.split(".")
        if (int(major), int(minor)) >= (0, 18):
            logger.info("UCM patching vllm for load-failure recovery...")
            import ucm.integration.vllm.patch.load_failure_patch

        if (int(major), int(minor)) >= (0, 23):
            logger.info("UCM patching vllm Model Runner V2 preemption order...")
            import ucm.integration.vllm.patch.v0230.vllm.v2_preemption_order_patch

        if (int(major), int(minor)) >= (0, 24):
            logger.info(
                "UCM patching vllm for EngineCore startup-failure worker shutdown..."
            )
            import ucm.integration.vllm.patch.engine_core_startup_shutdown_patch

        # vllm_ascend patches
        # Disable CpuAlloc.bind_memory BEFORE any cpu_binding_patch so that
        # bind_memory is a no-op before bind_threads replacement is installed.
        logger.info("UCM patching vllm-ascend bind_memory to no-op...")
        import ucm.integration.vllm.patch.bind_memory_patch

        # The CPU binding patch is version-independent since 0.26.0: the fixed
        # implementation wraps the upstream CpuAlloc methods (see
        # cpu_binding_affinity_patch.*_fixed), so every release >= 0.26.0 is
        # patched here; older versions keep their per-version cpu_binding
        # patches inside the match below. @when_imported is self-guarding
        # (only fires when vllm_ascend.cpu_binding exists).
        if ascend_version is not None and tuple(
            int(part) for part in ascend_version.split(".")
        )[:2] >= (0, 26):
            logger.info("UCM patching vllm-ascend CPU binding (fixed impl)...")
            import ucm.integration.vllm.patch.v0260.vllm_ascend.cpu_binding_patch

        match ascend_version:
            case "0.11.0":
                logger.info("UCM patching vllm-ascend for pc...")
                import ucm.integration.vllm.patch.v0110.vllm_ascend.pc_ascend_patch

                if ENABLE_SPARSE:
                    logger.info("UCM patching vllm-ascend for sparse...")
                    import ucm.integration.vllm.patch.v0110.vllm_ascend.sparse_ascend_patch
            case "0.18.0":
                logger.info("UCM patching vllm-ascend for pc...")
                import ucm.integration.vllm.patch.v0180.vllm_ascend.pc_ascend_patch
            case "0.17.0":
                logger.info(f"UCM patching vllm-ascend {ascend_version} for pc...")
                import ucm.integration.vllm.patch.v0180.vllm_ascend.ucm_connector_patch
            case "0.19.1":
                logger.info(f"UCM patching vllm-ascend {ascend_version} for pc...")
                import ucm.integration.vllm.patch.v0191.vllm_ascend.cpu_binding_patch
                import ucm.integration.vllm.patch.v0191.vllm_ascend.pc_ascend_patch
            case "0.20.2":
                logger.info(
                    "UCM patching vllm-ascend 0.20.2 for hybrid cache recovery..."
                )
                import ucm.integration.vllm.patch.v0202.vllm_ascend.ascend_hybrid_cache_patch
                import ucm.integration.vllm.patch.v0202.vllm_ascend.cpu_binding_patch
            case "0.21.0":
                logger.info(
                    "UCM patching vllm-ascend 0.21.0 for hybrid cache recovery..."
                )
                import ucm.integration.vllm.patch.v0210.vllm_ascend.ascend_hybrid_cache_patch
                import ucm.integration.vllm.patch.v0210.vllm_ascend.cpu_binding_patch
            case "0.22.1":
                logger.info(
                    "UCM patching vllm-ascend 0.22.1 for hybrid cache "
                    "recovery and CPU affinity..."
                )
                import ucm.integration.vllm.patch.v0221.vllm_ascend.ascend_hybrid_cache_patch
                import ucm.integration.vllm.patch.v0221.vllm_ascend.cpu_binding_patch
            case "0.23.0":
                logger.info(
                    "UCM patching vllm-ascend 0.23.0 for hybrid cache "
                    "recovery, CPU affinity, and SFA KV transfer..."
                )
                import ucm.integration.vllm.patch.v0230.vllm_ascend.ascend_hybrid_cache_patch
                import ucm.integration.vllm.patch.v0230.vllm_ascend.cpu_binding_patch

                if ENABLE_GSA_PATCH:
                    # GSA-aware SFA forward (drives
                    # maybe_execute_sparse_attention_begin/finished and the
                    # gsa_prefetch/warmup/swap top-k bookkeeping). Replaces
                    # the plain SFA KV-transfer forward below.
                    logger.info(
                        "UCM patching vllm-ascend 0.23.0 SFA with the GSA "
                        "forward (ENABLE_GSA_PATCH is set)..."
                    )
                    # GSA runtime monkey-patches (Python portions of
                    # vllm-ascend-adapt-1020-c8.patch). The C++ ops registered
                    # in csrc/torch_binding*.cpp still require the build-time
                    # patch vllm-ascend-csrc-build-1020-c8.patch to be applied
                    # before `pip install -v -e .` so the gsa_* ops are
                    # compiled in.
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm_ascend.block_table_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm_ascend.device_op_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm_ascend.model_runner_v1_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm_ascend.sfa_kv_transfer_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm_ascend.sfa_v1_patch
                    import ucm.sandbox.sparse.gsav2.patch.v0230.vllm_ascend.worker_patch
                else:
                    # Plain SFA KV-transfer forward (no GSA hooks).
                    logger.info(
                        "UCM patching vllm-ascend 0.23.0 SFA with the plain "
                        "KV-transfer forward (ENABLE_GSA_PATCH is not set)..."
                    )
                    import ucm.integration.vllm.patch.v0230.vllm_ascend.sfa_kv_transfer_patch
            case "0.24.0":
                logger.info(
                    "UCM patching vllm-ascend 0.24.0 for hybrid cache "
                    "recovery and CPU affinity..."
                )
                import ucm.integration.vllm.patch.v0240.vllm_ascend.ascend_hybrid_cache_patch
                import ucm.integration.vllm.patch.v0240.vllm_ascend.cpu_binding_patch
            case "0.25.1":
                logger.info(
                    "UCM patching vllm-ascend 0.25.1 for hybrid cache "
                    "recovery and CPU affinity..."
                )
                import ucm.integration.vllm.patch.v0251.vllm_ascend.ascend_hybrid_cache_patch
                import ucm.integration.vllm.patch.v0251.vllm_ascend.cpu_binding_patch
            case "0.30.0":
                logger.info("UCM patching vllm-ascend for chunked-prefill alignment...")
                import ucm.integration.vllm.patch.v0300.vllm_ascend.mamba_block_aligned_split_patch
            case "0.32.0":
                logger.info("UCM patching vllm-ascend for chunked-prefill alignment...")
                import ucm.integration.vllm.patch.v0300.vllm_ascend.mamba_block_aligned_split_patch
            case _:
                pass

        # Both release lines can omit the synchronous-load flag in Ascend's
        # copied balance scheduler. Reuse the idempotent metadata-driven fix;
        # do not enable unrelated v0300 patches on the v0320 release line.
        if ascend_version in {"0.30.0", "0.32.0"}:
            import ucm.integration.vllm.patch.v0300.vllm_ascend.balance_sync_kv_load_patch

        if ascend_version and tuple(map(int, ascend_version.split(".")[:2])) >= (0, 26):
            import ucm.integration.vllm.patch.v0260.vllm_ascend.minimax_m3_kv_transfer_patch

        # Fix: vllm-ascend >= 0.21.0 defers do_mamba_copy_block to after
        # start_load_kv, overwriting UCM-loaded data. @when_imported is
        # self-guarding (only fires when the module exists).
        import ucm.integration.vllm.patch.v0210.vllm_ascend.mamba_copy_order_patch
        import ucm.integration.vllm.patch.v0271.vllm.minimax_m3_kv_transfer_patch

        if (int(major), int(minor)) >= (0, 30):
            import ucm.integration.vllm.patch.v0300.multi_connector_handshake_patch

            # Apply the handshake patch during startup, after the other hooks
            # are registered for MultiConnector's transitive imports.
            import_module(
                "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
            )

        logger.info("UCM patch initialization completed!")

    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}\n")
        raise
