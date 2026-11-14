"""Connector v2 model-check for chunked prefill and external prefix hits."""

import argparse
import json
import os
import re
from contextlib import ExitStack, nullcontext
from math import lcm
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4


def token_count(value):
    if value.lower().endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="local model directory or its config.json",
    )
    p.add_argument(
        "--platform",
        choices=("cuda-sim", "ascend-sim", "cuda", "ascend"),
        required=True,
    )
    p.add_argument(
        "--kv-device",
        choices=("cpu", "cuda", "npu"),
        default="cpu",
        help="memory used for KV and Store readback (default: cpu)",
    )
    p.add_argument("--cuda-capability", metavar="MAJOR.MINOR")
    p.add_argument("--ascend-soc", help="installed vllm-ascend SoC family, e.g. A2")
    p.add_argument(
        "--tokens",
        type=token_count,
        default=32 * 1024,
        help="prompt length in tokens (default 32K; 1K = 1024)",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=token_count,
        default=8192,
        help="native scheduler token budget per step (default 8K)",
    )
    p.add_argument(
        "--assume-hit-percent",
        type=int,
        default=0,
        metavar="0..100",
        help="assumed external prefix hit in plan-only mode (default 0)",
    )
    p.add_argument(
        "--expect-policy",
        nargs="+",
        choices=("FA", "WA", "STATE"),
        default=[],
        help="require checked Dump windows on a cold run or Get windows on a hit",
    )
    p.add_argument(
        "--block-size",
        type=int,
        help="native vLLM block size (platform default if omitted)",
    )
    p.add_argument(
        "--kv-mib",
        type=int,
        help="KV pool budget in MiB (default: size one request with the platform KV planner)",
    )
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--pcp", type=int, default=1, help="prefill context parallel size")
    p.add_argument("--dcp", type=int, default=1, help="decode context parallel size")
    p.add_argument("--dtype", default="auto")
    p.add_argument(
        "--quantization", help="weight quantization used by the target model"
    )
    p.add_argument("--kv-cache-dtype", default="auto", help="target KV cache dtype")
    p.add_argument("--indexer-kv-dtype", default="auto", help="sparse indexer KV dtype")
    p.add_argument(
        "--mamba-ssm-cache-dtype",
        default="auto",
        help="Mamba/State cache dtype used by the official Kimi recipes",
    )
    p.add_argument(
        "--speculative-config",
        type=json.loads,
        help="JSON MTP/DSpark config; registers draft KV but does not run forward",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument(
        "--chunk-size", type=int, help="override UCM chunk size for range checking"
    )
    modes.add_argument(
        "--ucm-config",
        type=Path,
        help="also Dump and read back with this Store v2 YAML and matching DMA build",
    )
    args = p.parse_args()
    if args.model.name == "config.json":
        args.model = args.model.parent
    if not (args.model / "config.json").is_file():
        p.error("--model must point to a local model containing config.json")
    if args.ucm_config and not args.ucm_config.is_file():
        p.error("--ucm-config must be an existing file")
    if not 0 <= args.assume_hit_percent <= 100:
        p.error("--assume-hit-percent must be in [0, 100]")
    if args.ucm_config and args.assume_hit_percent:
        p.error("--assume-hit-percent requires plan-only mode (omit --ucm-config)")
    if args.speculative_config is not None:
        if args.speculative_config.get("method") not in (
            "mtp",
            "deepseek_mtp",
            "qwen3_5_mtp",
            "dspark",
        ):
            p.error("--speculative-config supports only MTP or DSpark")
        if args.ucm_config:
            p.error("Speculative model inspection supports plan-only mode")
        if args.speculative_config.get("method") == "dspark":
            # DeepSeek-V4/V4.1 ship their DSpark stages inside the target
            # checkpoint, so a draft directory is optional there. Kimi-K3 uses
            # a separate draft checkpoint and must name it.
            draft = args.speculative_config.get("model")
            if draft and not (Path(draft) / "config.json").is_file():
                p.error("DSpark draft must be a local model directory")
    sizes = (
        args.tokens,
        args.max_num_batched_tokens,
        args.kv_mib,
        args.block_size,
        args.chunk_size,
    )
    if any(size is not None and size <= 0 for size in sizes):
        p.error("token, block, chunk and memory sizes must be positive")
    if min(args.tp, args.pcp, args.dcp) < 1:
        p.error("--tp, --pcp and --dcp must be positive")
    if (args.pcp == 1 and args.tp % args.dcp) or (
        args.pcp > 1 and args.dcp not in (1, args.pcp, args.tp * args.pcp)
    ):
        p.error("DCP must divide TP with PCP=1, or equal 1, PCP or TP × PCP with PCP>1")
    if args.ucm_config and args.tp * args.pcp > 1:
        p.error("Multi-rank model-check supports plan-only mode; omit --ucm-config")
    if args.platform == "cuda-sim":
        if args.ascend_soc or not re.fullmatch(
            r"[1-9][0-9]*\.[0-9]", args.cuda_capability or ""
        ):
            p.error("cuda-sim requires --cuda-capability, e.g. 9.0")
    elif args.platform == "ascend-sim" and (
        args.cuda_capability or not args.ascend_soc
    ):
        p.error("ascend-sim requires --ascend-soc, e.g. A2")
    elif args.platform in ("cuda", "ascend") and (
        args.cuda_capability or args.ascend_soc
    ):
        p.error(
            "Real platforms discover the CUDA capability or Ascend SoC from the device"
        )
    required_platform = {"cuda": "cuda", "npu": "ascend"}.get(args.kv_device)
    if required_platform and args.platform != required_platform:
        p.error(f"--kv-device {args.kv_device} requires --platform {required_platform}")
    return args


def schedule_step(scheduler, req, cursor, hit, first, args):
    # Native schedule consumes and clears State hand-offs. Observe the original
    # result so expected checkpoint pages come directly from the allocator.
    manager = scheduler.kv_cache_manager
    take = manager.take_boundary_state_offloads
    offloads = []

    def observe_offloads():
        handoffs = take()
        offloads.extend(handoffs.get(req.request_id, ()))
        return handoffs

    with patch.object(manager, "take_boundary_state_offloads", observe_offloads):
        output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens.get(req.request_id, 0)
    if not 0 < scheduled <= args.max_num_batched_tokens:
        raise AssertionError(
            f"Invalid step budget or no progress: {output.num_scheduled_tokens}"
        )
    end = req.num_computed_tokens
    compute_start = end - scheduled
    replay = cursor - compute_start
    if (
        end > args.tokens
        or replay < 0
        or (replay and (not first or replay > scheduler.prefix_replay_tokens))
        or (replay and req.replay_start != compute_start)
    ):
        raise AssertionError(
            f"Unexpected step: prior={cursor}, scheduled={scheduled}, "
            f"compute_start={compute_start}, end={end}, replay={replay}"
        )
    stats = req.prefill_stats
    if (
        first
        and stats is not None
        and (stats.num_local_cached_tokens, stats.num_external_cached_tokens)
        != (0, hit)
    ):
        raise AssertionError(f"Unexpected native prefix-cache decision: {stats}")
    meta = output.kv_connector_metadata.requests[req.request_id]
    return output, (
        compute_start,
        end,
        scheduled,
        meta,
        manager.get_block_ids(req.request_id),
        offloads,
    )


def run(args, cleanup, scratch, kv_device):
    from native_cache import allocate, make_engine

    engine = make_engine(args)
    cfg, sched_cfg, block, hash_size, views = allocate(
        engine, args.kv_mib, scratch, cleanup, kv_device=kv_device
    )
    # Import the runtime entry points after the platform has installed its patches.
    import torch.distributed as dist
    from parallel import broadcast, inspect_workers
    from plans import assume_prefix, check_batches, unopened_clients
    from ranges import expected_records, fill_kv
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import make_empty_encoder_model_runner_output
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

    from ucm.integration.vllm.v2.runtime import make_common_ucm_chunk_ids

    root = dist.get_rank() == 0
    with nullcontext() if args.ucm_config else unopened_clients():
        if root:
            scheduler = engine.scheduler_config.get_scheduler_cls()(
                vllm_config=engine,
                kv_cache_config=sched_cfg,
                structured_output_manager=StructuredOutputManager(engine),
                block_size=block,
                hash_block_size=hash_size,
            )
            cleanup.callback(scheduler.shutdown)
        worker = KVConnectorFactory.create_connector(
            engine, KVConnectorRole.WORKER, cfg
        )
        if args.ucm_config:
            cleanup.callback(worker.shutdown)
        if kv_device.type == "cpu":
            worker.connector_worker._device = None
        worker.register_kv_caches(views)
        handshakes, placements = inspect_workers(worker, cfg, args)
        if root:
            scheduler.connector.set_xfer_handshake_metadata_pp_aware(handshakes)
    kv_buffers = fill_kv(views) if args.ucm_config else []
    layout = worker.connector_worker.layout
    chunk = layout.spec.chunk_size
    if args.tokens < chunk:
        raise ValueError(f"Use --tokens >= {chunk} for a complete UCM chunk")
    alignment = lcm(chunk, sched_cfg.cache_hit_alignment_tokens)
    requested_hit = args.tokens * args.assume_hit_percent // 100
    hit = min(requested_hit, args.tokens - 1) // alignment * alignment
    if args.assume_hit_percent and not hit:
        raise ValueError(f"Requested hit rounds to zero at alignment {alignment}")

    hashes = None
    if root:
        hash_fn = get_hash_fn_by_name(engine.cache_config.prefix_caching_hash_algo)
        init_none_hash(hash_fn)
        salt = uuid4().hex
        vocab_size = engine.model_config.get_vocab_size()
        req = Request(
            request_id=salt,
            prompt_token_ids=[i % vocab_size for i in range(args.tokens)],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            cache_salt=salt,
            block_hasher=get_request_block_hasher(hash_size, hash_fn),
        )
        role = scheduler.connector.connector_scheduler
        hashes = make_common_ucm_chunk_ids(role._chunk_hasher(req))
        if not args.ucm_config:
            assume_prefix(role, hashes, hit // chunk)
        scheduler.add_request(req)
    hashes = broadcast(hashes)
    print(
        f"prompt={args.tokens}, max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"chunk={chunk}, external_hit={hit} ({args.assume_hit_percent}% requested, "
        f"alignment={alignment}, actual={100 * hit / args.tokens:.2f}%), "
        f"fresh_prefill={args.tokens - hit}"
    )

    steps = num_get = num_dump = 0
    covered_policies = set()
    cursor = hit
    while cursor < args.tokens:
        step = None
        if root:
            output, step = schedule_step(scheduler, req, cursor, hit, steps == 0, args)
        compute_start, end, scheduled, meta, tables, offloads = broadcast(step)
        print(
            f"rank {dist.get_rank()} step {steps + 1}: "
            f"compute [{compute_start}, {end}), scheduled={scheduled}"
        )
        geometry = layout, cfg, views, tables, placements
        dumps = expected_records(
            *geometry, max(cursor, hit), end, state_offloads=offloads, hit_tokens=hit
        )
        if meta.prepare_get_failed or meta.prefetch_job:
            raise AssertionError("Unexpected Get preparation failure or prefetch")
        loads = checkpoints = {}
        if steps == 0 and hit:
            loads = expected_records(*geometry, 0, hit, state_load_target=end)
            checkpoints = expected_records(
                *geometry, 0, hit, state_load_target=hit, only_state=True
            )
        check_batches(
            worker.connector_worker,
            meta.load_plans,
            loads,
            hashes,
            is_dump=False,
            checkpoints=checkpoints,
        )
        check_batches(
            worker.connector_worker, meta.dump_plans, dumps, hashes, is_dump=True
        )
        covered_policies.update(policy.name for policy, _ in (loads if hit else dumps))
        print(f"step {steps + 1}: Get windows={len(loads)}, Dump windows={len(dumps)}")
        if root:
            model_output = make_empty_encoder_model_runner_output(output)
            if args.ucm_config:
                from dump import dump_and_check

                dump_and_check(
                    engine,
                    scheduler,
                    worker,
                    output,
                    model_output,
                    req.request_id,
                    views,
                    dumps,
                    kv_buffers,
                    kv_device,
                )
            else:
                scheduler.update_from_output(output, model_output)
        num_get += len(loads)
        num_dump += len(dumps)
        steps += 1
        cursor = end

    if root:
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
        if args.ucm_config:
            from dump import finish_store_work

            finish_store_work(engine, scheduler, worker)
        else:
            scheduler.schedule()
    if not (num_get or num_dump):
        raise AssertionError(
            "No complete Get or Dump window was checked for this request"
        )
    missing = set(args.expect_policy) - covered_policies
    if missing:
        op = "Get" if hit else "Dump"
        raise AssertionError(
            f"No checked {op} windows for {sorted(missing)}; observed {sorted(covered_policies)}"
        )
    return steps, num_get, num_dump


def main():
    args = arguments()
    print(f"Model-check arguments: {vars(args)}", flush=True)
    from parallel import launch

    if launch(args):
        return
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    import torch

    kv_device = torch.device("cpu")

    if args.platform == "cuda-sim":
        from host_runtime import install

        install("cuda", args.cuda_capability)
    elif args.platform == "ascend-sim":
        from host_runtime import install

        install("npu", args.ascend_soc.upper())
    else:
        if args.platform == "ascend":
            import torch_npu  # noqa: F401

        from vllm.platforms import current_platform

        device_type = "npu" if args.platform == "ascend" else "cuda"
        if current_platform.device_type != device_type:
            raise RuntimeError(
                f"Expected {device_type}, vLLM selected {current_platform.device_type}"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        accelerator = torch.device(device_type, local_rank)
        current_platform.set_device(accelerator)
        if args.kv_device != "cpu":
            kv_device = accelerator
    with TemporaryDirectory() as scratch, ExitStack() as cleanup:
        steps, gets, dumps = run(args, cleanup, scratch, kv_device)
    if int(os.environ.get("RANK", "0")) != 0:
        return
    mode = (
        "Dump + published-byte readback"
        if args.ucm_config
        else "plan and range check (no Store IO)"
    )
    print(
        f"PASS: {mode}, TP={args.tp} PCP={args.pcp} DCP={args.dcp}, {args.tokens} prompt tokens, {steps} steps, {gets} Get / {dumps} Dump windows"
    )


if __name__ == "__main__":
    main()
