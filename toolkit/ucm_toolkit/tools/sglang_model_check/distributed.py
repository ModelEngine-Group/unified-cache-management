"""Single-node TP/PP structure, device communication and CPU storage probe.

Launched by torchrun, not an inference server. Production SGLang decides layer
ownership and writer policy; production UCM decides physical storage identity.
"""

from __future__ import annotations

import inspect
import os
import signal
import traceback
import uuid
from dataclasses import replace
from datetime import timedelta

from .config import CheckConfig
from .result import CheckResult, emit_result
from .runner import (
    CheckFailure, _check_meta_model, _fill_pattern, _make_host_pool,
    _make_storage_config, _model_facts, _pool_page_views, _make_mamba_host_pool, _pool_name, _native_cache_dtype,
)
from .compatibility import requirements_from_facts
from .pool_probe import PoolSpec, build_pool_specs, checkpoint_pages, persisted_positions, expected_prefixes
from .native_probe import native_family, build_native_pools


def rank_coordinates(rank, tp_size, pp_size):
    if min(tp_size, pp_size) < 1 or not 0 <= rank < tp_size * pp_size:
        raise ValueError("invalid TP/PP rank or size")
    return rank % tp_size, rank // tp_size


def validate_partitions(ranges, total, tp_size, pp_size):
    """Require all TP replicas of each stage to agree, and PP to cover once."""
    if len(ranges) != tp_size * pp_size:
        raise ValueError("missing rank layer ranges")
    end = 0
    for stage in range(pp_size):
        owned = ranges[stage * tp_size:(stage + 1) * tp_size]
        start, stop = owned[0]
        if any(item != owned[0] for item in owned) or start != end or stop <= start:
            raise ValueError(f"invalid/inconsistent layer ownership at PP stage {stage}: {owned}")
        end = stop
    if end != total:
        raise ValueError(f"PP covers {end} layers, expected {total}")


def require_supported_model(facts):
    if facts.get("attention_arch") not in {"MHA", "MLA"} or any(
        facts.get(name) not in (True, False) for name in
        ("is_dsa", "is_deepseek_v4", "is_minimax_sparse")
    ) or any(facts.get(name) not in (True, False) for name in ("has_linear_attention", "is_hybrid_swa")):
        raise CheckFailure("CHECKER_RUNTIME_POOL_PROBE_REQUIRED", "distributed_scope",
                           "SGLang could not identify a supported attention/cache family for TP/PP probing")


def _phase(dist, control, world_size, name, action):
    """Share failures before any rank enters the next collective phase."""
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = (getattr(exc, "code", "ENVIRONMENT_UNSUPPORTED"),
                 f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    errors = [None] * world_size
    dist.all_gather_object(errors, error, group=control)
    failures = [e for e in errors if e]
    if failures:
        raise CheckFailure(failures[0][0], name, "\n".join(e[1] for e in failures))
    return value


def _communication(torch, groups, device):
    dist = torch.distributed
    for group in groups:
        ranks = group.ranks
        if len(ranks) == 1:
            continue
        tensor = torch.tensor([dist.get_rank() + 1], dtype=torch.int64, device=device)
        dist.all_reduce(tensor, group=group.device_group)
        if tensor.item() != sum(rank + 1 for rank in ranks):
            raise RuntimeError("TP/PP all-reduce mismatch")
        tensor.fill_(dist.get_rank())
        dist.broadcast(tensor, src=ranks[0], group=group.device_group)
        if tensor.item() != ranks[0]:
            raise RuntimeError("TP/PP broadcast mismatch")
    # Exercise PP point-to-point links as well as collective communication.
    pp = groups[1]
    if len(pp.ranks) == 1:
        return
    position = pp.ranks.index(dist.get_rank())
    sent = torch.tensor([dist.get_rank()], device=device, dtype=torch.int64)
    received = torch.empty_like(sent)
    pending = []
    if position + 1 < len(pp.ranks):
        pending.append(dist.isend(sent, dst=pp.ranks[position + 1], group=pp.device_group))
    if position:
        pending.append(dist.irecv(received, src=pp.ranks[position - 1], group=pp.device_group))
    for request in pending:
        request.wait()
    if position and received.item() != pp.ranks[position - 1]:
        raise RuntimeError("PP point-to-point mismatch")


def _probe(config, result):
    import torch
    import torch.distributed as dist
    from sglang.srt.configs.model_config import ModelConfig, is_deepseek_v4, is_deepseek_dsa, get_dsa_index_kpool
    from sglang.srt.distributed import parallel_state
    from sglang.srt.model_executor.model_runner_components.layer_setup import resolve_layer_indices
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.runtime_context import SpawnRanks, get_server_args, publish
    from .sglang_compat import environment_info, refine_facts_from_model, server_args_guard
    from ucm.integration.sglang.unifiedcache_store import UnifiedCacheStore

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world_size = config.tp_size * config.pp_size
    if int(os.environ["WORLD_SIZE"]) != world_size or int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
        raise ValueError("TP/PP validation requires a single node and WORLD_SIZE=TP*PP")
    tp_rank, pp_rank = rank_coordinates(rank, config.tp_size, config.pp_size)
    result.environment = environment_info()
    platform = result.environment["platform"]
    if config.platform != "auto" and config.platform != platform:
        raise ValueError(f"requested {config.platform}, detected {platform}")
    if platform == "ascend":
        import torch_npu  # noqa: F401 - registers the NPU device/backend

        torch.npu.set_device(local_rank)
        device, backend = f"npu:{local_rank}", "hccl"
    elif platform in {"cuda", "rocm"}:
        torch.cuda.set_device(local_rank)
        device, backend = f"cuda:{local_rank}", "nccl"
    else:
        raise ValueError("TP/PP device probe requires CUDA/ROCm/Ascend")
    model_config = ModelConfig(model_path=config.model, dtype=config.dtype,
                               trust_remote_code=config.trust_remote_code)
    dsv4 = is_deepseek_v4(model_config.hf_config)
    if config.page_size == 0:
        page_size = 256 if dsv4 else 64
        if is_deepseek_dsa(model_config.hf_config):
            page_size = (1 if platform == "rocm" else 64) * get_dsa_index_kpool(model_config.hf_config)
        config = replace(config, page_size=page_size)
    cache_fields = dict(kv_cache_dtype=config.kv_cache_dtype)
    if dsv4:
        cache_fields.update(dsv4_attn_backend=config.dsv4_attn_backend,
                            enable_deepseek_v4_fp4_indexer=config.fp4_indexer,
                            c128_page_size=config.c128_page_size)
    host, storage = None, None
    native_bundle = None
    pools = {}
    with server_args_guard(single_rank=False, model_path=config.model,
                           tp_size=config.tp_size, pp_size=config.pp_size,
                           dp_size=1, attn_dp_size=1, attn_cp_size=1, dcp_size=1,
                           moe_dp_size=1, ep_size=1,
                           disable_custom_all_reduce=True, page_size=config.page_size or 64,
                           hicache_mem_layout=config.layout, **cache_fields):
        try:
            # Publish launcher placement as SGLang does, including gpu_id and
            # derived ranks that individual model constructors may consult.
            publish(get_server_args(), role="scheduler",
                    ranks=SpawnRanks(world_rank=rank, dp_rank=0, gpu_id=local_rank))
            parallel_state.init_distributed_environment(world_size=world_size, rank=rank,
                local_rank=local_rank, backend=backend, timeout=config.distributed_timeout)
            kwargs = dict(backend=backend, tensor_model_parallel_size=config.tp_size,
                          pipeline_model_parallel_size=config.pp_size)
            parameters = inspect.signature(parallel_state.initialize_model_parallel).parameters
            parallel_state.initialize_model_parallel(**{k: v for k, v in kwargs.items() if k in parameters})
            control = dist.new_group(backend="gloo", timeout=timedelta(seconds=config.distributed_timeout))
            phase = lambda name, action: _phase(dist, control, world_size, name, action)
            groups = (parallel_state.get_tp_group(), parallel_state.get_pp_group())
            phase("device_communication", lambda: _communication(torch, groups, device))
            result.validation["device_communication"] = "passed"

            def describe():
                facts = _model_facts(model_config)
                require_supported_model(facts)
                model = _check_meta_model(model_config, use_current_context=True)
                facts = refine_facts_from_model(facts, model)
                require_supported_model(facts)
                if facts.get("is_hybrid_swa"):
                    try:
                        from sglang.srt.model_executor.model_runner_components.load_model_utils import resolve_sliding_window_size
                    except ImportError:
                        resolver = getattr(model, "get_attention_sliding_window_size", None)
                        if callable(resolver):
                            model_config.sliding_window_size = resolver()
                    else:
                        model_config.sliding_window_size = resolve_sliding_window_size(model, model_config)
                args = dict(model=model, model_config=model_config, is_draft_worker=False,
                            spec_algorithm=SpeculativeAlgorithm.NONE)
                params = inspect.signature(resolve_layer_indices).parameters
                info = resolve_layer_indices(**{k: v for k, v in args.items() if k in params})
                return model_config, facts, (int(info.start_layer), int(info.end_layer))

            model_config, facts, owned = phase("meta_model_init", describe)
            result.model_info = facts
            ranges = [None] * world_size
            dist.all_gather_object(ranges, owned, group=control)
            phase("pp_layer_partition", lambda: validate_partitions(
                ranges, model_config.num_hidden_layers, config.tp_size, config.pp_size))
            result.validation["meta_model"] = "passed"
            requirements = requirements_from_facts(facts, config.layout)
            family = native_family(requirements)
            result.cache_requirements = requirements.to_dict()

            def allocate():
                nonlocal host, storage, native_bundle, pools
                if config.layout not in {"page_first", "page_first_direct"} or (not family and config.kv_cache_dtype != "auto"):
                    raise ValueError("TP/PP requires page-first layouts; explicit KV dtype requires a native provider")
                if family:
                    native_bundle = build_native_pools(model_config, requirements, config, platform,
                        _native_cache_dtype(model_config, config, family, platform),
                        layer_range=owned, tp_size=config.tp_size)
                    pools = native_bundle.pools
                    specs = native_bundle.specs
                elif requirements.is_hybrid:
                    specs = build_pool_specs(model_config, requirements.pool_names, config.page_size,
                                             platform, tp_size=config.tp_size, layer_range=owned)
                elif requirements.is_mla:
                    spec = PoolSpec("kv", "mla", tuple(range(*owned)),
                                    kv_cache_dim=model_config.kv_lora_rank + model_config.qk_rope_head_dim)
                    specs = [spec]
                else:
                    spec = PoolSpec("kv", "mha", tuple(range(*owned)),
                        head_num=model_config.get_num_kv_heads(config.tp_size),
                        head_dim=model_config.head_dim, v_head_dim=model_config.v_head_dim)
                    specs = [spec]
                estimate = sum(spec.bytes_per_slot(model_config.dtype, config.page_size)
                               for spec in specs) * (config.pages * 2 + 2) * 8
                if not family and requirements.is_hybrid and estimate > config.max_host_mib * 1024 * 1024:
                    raise CheckFailure("CHECKER_MEMORY_LIMIT", "pool_registration", "per-rank hybrid pool budget exceeded")
                for spec in specs:
                    if family:
                        continue  # Native producer already built the physical Host pools.
                    if spec.kind == "logical":
                        from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
                        pool = LogicalHostPool((config.pages * 2 + 2) * config.page_size,
                                               config.page_size, layout=config.layout)
                    elif spec.kind == "mamba":
                        pool = _make_mamba_host_pool(spec, config)
                    else:
                        pool = _make_host_pool(model_config, requirements, config, spec,
                                              layer_start=0 if requirements.is_hybrid else owned[0])
                    pools[spec.name] = pool
                host = pools["kv"]
                cfg = _make_storage_config(config, requirements)
                cfg.tp_rank, cfg.tp_size = tp_rank, config.tp_size
                cfg.pp_rank, cfg.pp_size = pp_rank, config.pp_size
                # SGLang treats DeepSeek V4's logical KV and side pools as
                # rank-replicated MLA, regardless of ModelConfig's MHA label.
                cfg.is_mla_model = requirements.is_mla or family == "deepseek_v4"
                storage = UnifiedCacheStore(cfg)
                storage.register_mem_pool_host(host)
                for spec in specs:
                    if spec.name != "kv":
                        storage.register_mem_host_pool_v2(pools[spec.name], _pool_name(spec.name))
                return cfg, {spec.name: spec for spec in specs}

            storage_config, specs = phase("pool_registration", allocate)
            token_count = config.pages * host.page_size
            source_indices = torch.arange(token_count, dtype=torch.int64)
            target_indices = source_indices + token_count
            salt = [uuid.uuid4().hex if rank == 0 else None]
            dist.broadcast_object_list(salt, src=0, group=control)
            keys = [f"sglang-tp-pp-check-{salt[0]}-{page}" for page in range(config.pages)]
            snapshots = {}
            written_side_pools = set()
            windows = {name: spec.window_pages for name, spec in specs.items()}
            positions = {name: set(persisted_positions(config.pages, windows[name], checkpoint_pages(config.pages)))
                         for name in specs}
            from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolHitPolicy

            def transfers(target=False, query=None, placeholders=False):
                result = []
                for name in specs:
                    if name == "kv":
                        continue
                    if query is not None:
                        tail = query[-windows[name]:] if windows[name] else query
                        values = ["__placeholder__"] * len(tail) if placeholders and windows[name] else tail
                        indices = None
                    else:
                        pages = sorted(positions[name])
                        values = [keys[page] for page in pages]
                        size = pools[name].page_size
                        indices = torch.tensor([i + (config.pages * size if target else 0)
                            for page in pages for i in range(page * size, (page + 1) * size)], dtype=torch.int64)
                    result.append(PoolTransfer(name=_pool_name(name), keys=values, host_indices=indices,
                        hit_policy=PoolHitPolicy.TRAILING_PAGES if windows[name] else PoolHitPolicy.ALL_PAGES))
                return result

            def prepare():
                for pool_number, (name, spec) in enumerate(specs.items()):
                    if spec.kind == "logical":
                        continue
                    for page in positions[name]:
                        source = _pool_page_views(pools[name], name, page, requirements.is_mla)
                        target = _pool_page_views(pools[name], name, config.pages + page, requirements.is_mla)
                        for component, (src, dst) in enumerate(zip(source, target)):
                            replicated = storage_config.is_mla_model and name != "mamba"
                            owner = pp_rank * config.tp_size + (0 if replicated else tp_rank)
                            seed = owner * 1000003 + pool_number * 100003 + page * 101 + component
                            _fill_pattern(src, seed=seed)
                            snapshots[name, page, component] = src.clone()
                            dst.zero_()

            def require_results(values):
                if len(values) != config.pages or not all(values):
                    raise CheckFailure("STORAGE_IO_FAILED", "storage_io", f"storage returned {values!r}")

            def query_consensus(query, expected):
                def lookup():
                    return (storage.batch_exists_v2(query, transfers(query=query, placeholders=True)).kv_hit_pages
                            if requirements.is_hybrid else storage.batch_exists(query))
                local_hits = phase("lookup", lookup)
                hit_tensor = torch.tensor(local_hits, dtype=torch.int64)
                dist.all_reduce(hit_tensor, op=dist.ReduceOp.MIN, group=control)
                expected_tensor = torch.tensor(max(expected) if expected else 0, dtype=torch.int64)
                dist.all_reduce(expected_tensor, op=dist.ReduceOp.MIN, group=control)
                def check():
                    actual = hit_tensor.item()
                    if actual != expected_tensor.item() or (actual and actual not in expected):
                        raise CheckFailure("LOOKUP_CONTRACT_MISMATCH", "lookup_consensus",
                            f"MIN hit={actual}, expected MIN={expected_tensor.item()}, local candidates={expected}")
                phase("lookup_consensus", check)

            phase("prepare_data", prepare)
            query_consensus(keys, [])
            # Older SGLang keeps this rule in the controller, not the config.
            # This probe disables DCP: MLA writes on TP0, MHA on every TP rank.
            writer = getattr(storage_config, "is_storage_writer",
                             not storage_config.is_mla_model or storage_config.tp_rank == 0)
            def dump():
                selected = transfers()
                if storage_config.is_mla_model and not writer and selected:
                    if "mamba" not in specs:
                        selected = []  # These supported MLA/DSV4 sidecars are replicated.
                    else:
                        from types import SimpleNamespace
                        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
                        policy = SimpleNamespace(backup_skip=True, storage_backend_type="unifiedcache")
                        selected = [t for t in selected if HybridCacheController.should_backup(policy, t)]
                if selected:
                    from .runner import _require_page_results
                    results = storage.batch_set_v2(selected)
                    for transfer in selected:
                        _require_page_results(results, [transfer.name], len(transfer.keys), "DUMP_FAILED", "dump")
                        written_side_pools.add(getattr(transfer.name, "value", transfer.name))
                if writer:
                    require_results(storage.batch_set_v1(keys, source_indices))

            phase("dump", dump)
            candidates = expected_prefixes(config.pages, positions, windows)
            query_consensus(keys, candidates)
            query_consensus([f"missing-{key}" for key in keys], [])
            if config.pages > 1:
                query_consensus(keys[:-1], expected_prefixes(config.pages - 1, positions, windows))

            def load_compare():
                require_results(storage.batch_get_v1(keys, target_indices))
                if requirements.is_hybrid:
                    from .runner import _require_page_results
                    selected = transfers(target=True)
                    results = storage.batch_get_v2(selected)
                    for transfer in selected:
                        _require_page_results(results, [transfer.name], len(transfer.keys), "LOAD_FAILED", "load_compare")
                for (name, page, component), expected in snapshots.items():
                    actual = _pool_page_views(pools[name], name, config.pages + page, requirements.is_mla)[component]
                    if not torch.equal(expected, actual):
                        raise CheckFailure("ROUNDTRIP_MISMATCH", "load_compare",
                                           f"rank data mismatch: pool={name}, page={page}, component={component}")

            phase("load_compare", load_compare)
            details = [None] * world_size
            dist.all_gather_object(details, dict(rank=rank, tp_rank=tp_rank, pp_rank=pp_rank,
                layer_range=owned, write="passed" if writer else "skipped_by_policy",
                read="passed", pool_specs={name: spec.report() for name, spec in specs.items()},
                side_writes={name: "passed" if name in written_side_pools else "skipped_by_policy"
                             for name in specs if name != "kv"},
                bytes_compared=sum(x.numel() * x.element_size() for x in snapshots.values())), group=control)
            result.roundtrip = {"ranks": details}
            if native_bundle is not None:
                result.roundtrip["native_provider"] = native_bundle.report
            result.validation.update(storage_io="passed", distributed_consistency="passed",
                                     runtime_pools="per_rank_host_pools")
            result.succeed("roundtrip", "TP/PP model structure, communication and UCM storage passed")
        finally:
            if storage is not None:
                storage.close()
            if native_bundle is not None:
                native_bundle.close()
            else:
                for pool in pools.values():
                    pool.destroy()
            if dist.is_initialized():
                parallel_state.cleanup_dist_env_and_memory()


def main():
    config = CheckConfig.from_env()
    rank = int(os.environ.get("RANK", "0"))
    result = CheckResult(model=config.model, mode=config.mode)
    result.validation = dict(scope="tp_pp_host_storage_contract", tp_size=config.tp_size,
        pp_size=config.pp_size, meta_model="not_run", device_communication="not_run",
        storage_io="not_run", inference="not_run", device_transfer="not_run")
    # torchrun terminates peers when a worker exits. Alarm also bounds non-
    # collective phases; process-group timeouts bound blocked collectives.
    def timeout(signum, frame):
        # Do not enter potentially blocking backend cleanup after a timeout.
        # A nonzero worker exit makes torchrun terminate its peers.
        os.write(2, b"TP/PP worker exceeded --distributed-timeout\n")
        os._exit(124)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config.distributed_timeout)
    try:
        _probe(config, result)
    except Exception as exc:
        result.fail(getattr(exc, "code", "ENVIRONMENT_UNSUPPORTED"),
                    getattr(exc, "stage", "distributed_setup"),
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        signal.alarm(0)
    if rank == 0:
        emit_result(result, config.output, verbose=config.verbose)
    return 0 if result.supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
