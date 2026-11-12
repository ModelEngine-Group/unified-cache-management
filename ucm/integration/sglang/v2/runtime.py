"""Thin access to both interfaces of one native Store v2 assembler.

The caller owns request/job lifetimes, serialization and completion consumption.
This module does not add locks, queues, polling or Python protocol state.
"""

from importlib import import_module
from pathlib import Path

import yaml


def proxy():
    return import_module("ucm_proxy")


class Runtime:
    def __init__(self, extra_config: dict):
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
        self.assembler = proxy().StoreAssembler(config)
        self.scheduler = self.assembler.get_scheduler()
        self.worker = self.assembler.get_worker()

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

    def complete_get(self, request_id):
        return self.scheduler.complete_get(request_id)

    def prepare_put(self, request_id, keys):
        return self.scheduler.prepare_put(request_id, keys)

    def complete_put(self, request_id, success):
        return self.scheduler.complete_put(request_id, success)

    def prepare_prefetch(self, request_id):
        return self.scheduler.prepare_prefetch(request_id)

    def complete_prefetch(self, request_id, success):
        return self.scheduler.complete_prefetch(request_id, success)

    def submit_get(self, job_id, payloads, token, context):
        return self.worker.submit_get(job_id, payloads, token, context)

    def submit_put(self, job_id, payloads, token, context):
        return self.worker.submit_put(job_id, payloads, token, context)

    def submit_prefetch(self, job_id, token):
        return self.worker.submit_prefetch(job_id, token)

    def get_finished(self):
        return self.worker.get_finished()

    def wait(self, job_id):
        return self.worker.wait(job_id)
