"""Thin access to both interfaces of one native Store v2 assembler.

The caller owns request/job lifetimes, serialization and completion consumption.
This module does not add locks, queues, polling or Python protocol state.
"""

from importlib import import_module
from pathlib import Path

import yaml

from ucm.logger import init_logger

logger = init_logger(__name__)


def proxy():
    return import_module("ucm_proxy")


class Runtime:
    def __init__(
        self,
        extra_config: dict,
        *,
        unique_id: str,
        block_size: int,
        device_id: int,
        dump_partitions: int = 1,
        coordinated_puts: bool = False,
    ):
        config_file = extra_config.get("UCM_CONFIG_FILE")
        if not config_file:
            raise ValueError(
                "SGLang UCM linker requires linker_extra_config.UCM_CONFIG_FILE."
            )
        with Path(config_file).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            raise ValueError("UCM configuration YAML root must be a mapping.")
        connectors = document.get("ucm_connectors")
        if not isinstance(connectors, list) or len(connectors) != 1:
            raise ValueError(
                "SGLang UCM linker requires exactly one ucm_connectors entry."
            )
        connector = connectors[0]
        config = (
            connector.get("ucm_connector_config")
            if isinstance(connector, dict)
            else None
        )
        if not isinstance(config, dict):
            raise ValueError("UCM connector entry is missing ucm_connector_config.")
        branches = config.get("branches")
        if coordinated_puts:
            pipe = (
                branches[0].get("pipe")
                if isinstance(branches, list)
                and len(branches) == 1
                and isinstance(branches[0], dict)
                else None
            )
            if (
                not isinstance(pipe, list)
                or not pipe
                or not isinstance(pipe[0], dict)
                or pipe[0].get("store") != "cache"
            ):
                raise ValueError("Sharded Store V2 Put requires one Cache-first branch")
        if isinstance(branches, list):
            for branch in branches:
                pipe = branch.get("pipe") if isinstance(branch, dict) else None
                if isinstance(pipe, list):
                    for tier in pipe:
                        if isinstance(tier, dict):
                            # Set the default per tier: the assembler rejects
                            # collisions between runtime and tier config keys.
                            options = tier.setdefault("config", {})
                            if isinstance(options, dict):
                                options.setdefault("llm_role", "both")
                                # Existing YAML may still carry these derived values.
                                options.pop("unique_id", None)
                                options.pop("block_size", None)
                                options.pop("device_id", None)
        self.assembler = proxy().StoreAssembler(
            config,
            {"unique_id": unique_id, "block_size": block_size, "device_id": device_id},
        )
        self.scheduler = self.assembler.get_scheduler()
        self.worker = self.assembler.get_worker()
        logger.info(
            "SGLang UCM V2 Store initialized: config_file=%s store_id=%s "
            "device_id=%s block_size_bytes=%s branches=%s",
            config_file,
            unique_id,
            device_id,
            block_size,
            [
                [
                    {
                        "store": tier["store"],
                        "llm_role": tier.get("config", {}).get("llm_role", "both"),
                        **(
                            {
                                "cache_buffer_capacity_gb": tier.get("config", {}).get(
                                    "cache_buffer_capacity_gb", "default"
                                )
                            }
                            if tier["store"] == "cache"
                            else {}
                        ),
                    }
                    for tier in branch["pipe"]
                ]
                for branch in branches
            ],
        )

    def on_new_request(self, context):
        return self.scheduler.on_new_request(context)

    def on_request_finished(self, request_id):
        return self.scheduler.on_request_finished(request_id)

    def lookup(self, request_id, keys):
        return self.scheduler.lookup(request_id, keys)

    def lookup_on_prefix(self, request_id, keys):
        return self.scheduler.lookup_on_prefix(request_id, keys)

    def lookup_on_reverse(self, request_id, keys):
        return self.scheduler.lookup_on_reverse(request_id, keys)

    def touch(self, request_id, keys):
        return self.scheduler.touch(request_id, keys)

    def prepare_get(self, request_id, keys):
        return self.scheduler.prepare_get(request_id, keys)

    def complete_get(self, request_id, token):
        return self.scheduler.complete_get(request_id, token)

    def prepare_put(self, request_id, keys):
        return self.scheduler.prepare_put(request_id, keys)

    def complete_put(self, request_id, token, success):
        return self.scheduler.complete_put(request_id, token, success)

    def prepare_offload(self):
        return self.scheduler.prepare_offload()

    def complete_offload(self, plan):
        return self.scheduler.complete_offload(plan)

    def prepare_prefetch(self, request_id):
        return self.scheduler.prepare_prefetch(request_id)

    def complete_prefetch(self, request_id, success):
        return self.scheduler.complete_prefetch(request_id, success)

    def submit_get(self, job_id, payloads, token, context):
        return self.worker.submit_get(job_id, payloads, token, context)

    def submit_put(self, job_id, payloads, token, context):
        return self.worker.submit_put(job_id, payloads, token, context)

    def submit_offload(self, job_id, plan):
        return self.worker.submit_offload(job_id, plan)

    def submit_prefetch(self, job_id, token):
        return self.worker.submit_prefetch(job_id, token)

    def get_finished(self):
        return self.worker.get_finished()

    def wait(self, job_id):
        return self.worker.wait(job_id)
