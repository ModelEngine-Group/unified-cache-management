"""Single owner protocol on the existing control channel and job table.

Credits cover confirmed prefix batches before their data work is published.
Candidate probing allocates no data slots; live claims remain until owner drain.
This ledger never substitutes for native pins or guesses process-death recovery.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid

from .early_control import BrokerState

log = logging.getLogger(__name__)
VERSION = 3


def worker_capability(connector):
    """Read the native object and this process's kernel identities after setup."""
    try:
        config = connector.store.config
        native = connector.store.domain_info()
        if not connector.launch_config.get("enable_early_owner", True):
            return {"reason": "owner explicitly disabled"}
        if not connector.is_mla or not config.get("share_buffer_enable"):
            return {"reason": "not native MLA shared L1"}
        if connector.cp_world_size != 1 or not native["backing"]:
            return {"reason": "unverified shared mapping or CP layout"}
        if getattr(connector._vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1:
            return {"reason": "PP consumer cohort is not supported by owner protocol"}
        shard, block = native["shard_bytes"], native["block_bytes"]
        if not shard or block % shard or shard != config["shard_size"] or block != config["block_size"]:
            return {"reason": "native/config layout mismatch"}
        layers = list(connector.layer_ids) if connector.use_layerwise else [0]
        if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= block // shard:
            return {"reason": "unsupported consumer shard range"}
        model_len = connector._vllm_config.model_config.max_model_len
        sequences = connector._vllm_config.scheduler_config.max_num_seqs
        if not isinstance(model_len, int) or model_len <= 0 or not isinstance(sequences, int) or sequences <= 0:
            return {"reason": "native maximum task range is not bounded"}
        # Bound queued lookup work, not memory reserved for hypothetical tasks.
        token_unit = min(connector.hash_block_size, connector.block_size)
        task_shards = (model_len + token_unit - 1) // token_unit
        if not connector.use_layerwise:
            task_shards *= block // shard
        dump_task_shards = task_shards * sequences if connector.tp_rank % connector.tp_size == 0 else 0
        # Actual object identity, physical host and namespace reachability.
        domain = [Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  os.readlink("/proc/self/ns/ipc"), os.readlink("/proc/self/ns/mnt"), native["backing"]]
        # Native workers have no scheduler _seed, and their own RequestHasher
        # includes the worker rank. MLA IDs arrive from the rank-0 scheduler
        # hasher, so fingerprint that same native identity once at handshake.
        scheduler_hasher = type(connector.request_hasher)(connector._vllm_config, 0)
        layout = [connector.unique_id, native, config["tensor_size_list"],
                  connector.use_layerwise, connector.hash_block_size, connector.blocks_per_chunk,
                  scheduler_hasher("UCM_HASH_SEED").hex()]
        return dict(version=VERSION, domain=json.dumps(domain),
                    layout=hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
                    unique_id=str(connector.unique_id), native=native, layers=layers,
                    task_shards=task_shards, dump_task_shards=dump_task_shards,
                    max_jobs=int(connector.launch_config.get("early_owner_max_jobs", 4)),
                    max_candidates=int(connector.launch_config.get("early_owner_max_candidate_blocks", 0)),
                    dp=connector._vllm_config.parallel_config.data_parallel_index,
                    rank=connector.tp_rank % connector.tp_size,
                    ready=True, pid=os.getpid())
    except Exception as error:
        return {"reason": f"capability unavailable: {type(error).__name__}: {error}"}


class OwnerBroker(BrokerState):
    def __init__(self, send, tp_size, timeout=300, dp_size=1):
        super().__init__(send, tp_size, timeout)
        self.dp_size = dp_size
        self.capabilities, self.sessions, self.domains = {}, {}, {}
        self.cursor, self.last_sequence = {}, {}
        self.fenced = False
        self.fence_reason = None
        self.next_grant = 0

    def cohort(self):
        return set().union(*(self.workers(dp) for dp in range(self.dp_size)))

    def compatible(self, dp):
        if self.fenced:
            return None, "shared cohort fenced; restart with a fresh native L1"
        cohort = self.cohort()
        if not cohort.issubset(self.peers) or any(time.monotonic() - self.peers[p] > 60 for p in cohort if p in self.peers):
            if self.domains:
                self.poison("shared cohort no longer healthy")
            return None, "incomplete healthy control cohort"
        if not cohort.issubset(self.capabilities):
            return None, "incomplete native capability handshake"
        for peer in cohort:
            cap = self.capabilities[peer]
            if cap.get("version") != VERSION or not cap.get("ready"):
                return None, cap.get("reason", "worker not ready")
            if peer != f"w{cap.get('dp')}.{cap.get('rank')}" or cap.get("max_jobs", 0) < 1:
                return None, "invalid worker membership/capacity"
            if cap.get("task_shards", 0) <= 0 or cap.get("dump_task_shards", -1) < 0:
                return None, "native maximum task range is not bounded"
        group = [self.capabilities[p] for p in sorted(self.workers(dp))]
        if len({(c["domain"], c["layout"]) for c in group}) != 1:
            return None, "DP consumers do not share one compatible native object"
        domain = group[0]["domain"]
        members = [p for p in cohort if self.capabilities[p]["domain"] == domain]
        if len({self.capabilities[p]["layout"] for p in members}) != 1:
            return None, "shared object has incompatible DP layouts"
        if any(j.get("running") and not j.get("owner") for j in self.jobs.values()):
            return None, "legacy reservations still belong to this cohort"
        if domain not in self.domains:
            native = group[0]["native"]
            # Native allocation enforces the actual shared capacity. Early load
            # and dump pre-acquire handles with a finite allocation bound and
            # reject before I/O, so no maximum-task subtraction is necessary.
            slots = native["reservable_nodes"]
            limit = slots * native["shard_bytes"]
            io = min(limit, len(members) * 128 * native["shard_bytes"])
            quota = io // len(members) // native["shard_bytes"] * native["shard_bytes"]
            self.domains[domain] = dict(limit=limit, used=0, peak=0, io_limit=io,
                                        io_quota=quota, members=members, owners={}, claims={},
                                        block_bytes=native["block_bytes"],
                                        native_load_headroom_bytes=(native["nodes"] - slots) * native["shard_bytes"],
                                        native_dump_headroom_bytes=0)
            for peer in self.peers:
                self.send(peer, dict(op="domain_active"))
        return self.domains[domain], None

    def select_owner(self, candidates, dp):
        """Policy extension point: eligibility and accounting stay outside it."""
        order = sorted(self.workers(dp))
        start = self.cursor.get(dp, 0)
        for offset in range(len(order)):
            index = (start + offset) % len(order)
            if order[index] in candidates:
                self.cursor[dp] = (index + 1) % len(order)
                return order[index]
        return None

    def dispatch(self, job):
        if job["status"] != "PENDING" or job.get("running"):
            return
        if not self.workers(job["dp"]).issubset(self.peers):
            return
        budget, reason = self.compatible(job["dp"])
        if budget is None:
            if self.fenced:
                self.fail(job, reason)
                return
            log.info("UCM_EARLY owner_disabled key=%s dp=%s reason=%s", job["key"], job["dp"], reason)
            return super().dispatch(job)
        work = len(job["blocks"])
        claims, charge = set(), 0
        candidates = []
        job_limited = candidate_limited = 0
        for peer in self.workers(job["dp"]):
            load = budget["owners"].get(peer, dict(jobs=0, candidates=0))
            peer_cap = self.capabilities[peer]
            work_limit = peer_cap["max_candidates"] or peer_cap["max_jobs"] * peer_cap["task_shards"]
            if load["jobs"] >= peer_cap["max_jobs"]:
                job_limited += 1
            elif load["candidates"] + work > work_limit:
                candidate_limited += 1
            elif peer in self.peers:
                candidates.append(peer)
        owner = self.select_owner(candidates, job["dp"])
        if owner is None:
            reason = "owner_work_limits"
            log.info("UCM_EARLY owner_nospace key=%s dp=%s candidates=%s protected=%s limit=%s "
                     "reason=%s incremental_bytes=%s job_limited=%s candidate_limited=%s wasted_read_bytes=0",
                     job["key"], job["dp"], work, budget["used"], budget["limit"],
                     reason, charge, job_limited, candidate_limited)
            self.fail(job, "bounded owner capacity: " + reason + "; native fallback")
            return
        load = budget["owners"].setdefault(owner, dict(jobs=0, candidates=0, assignments=0))
        load["jobs"] += 1
        load["candidates"] += work
        load["assignments"] += 1
        self.next_grant += 1
        job.update(owner=owner, attempt=uuid.uuid4().hex, grant=self.next_grant, running=True, budget=budget,
                   claims=claims, charge=0, confirmed=0, batch_grants={},
                   consumers=set(), terminal=set(), assigned_ns=time.monotonic_ns())
        envelope = dict(key=job["key"], attempt=job["attempt"], grant=job["grant"], owner=owner, dp=job["dp"],
                        request=job.get("request"), candidate_blocks=work,
                        submitted_ns=job.get("submitted_ns"), io_quota=budget["io_quota"])
        for peer in self.workers(job["dp"]):
            self.send(peer, dict(envelope, op="owner_reserve", blocks=job["blocks"]) if peer == owner
                      else dict(envelope, op="observe"))
        self.send(job["client"], dict(op="owner_assigned", key=job["key"],
                  owner=owner, attempt=job["attempt"], grant=job["grant"]))
        log.info("UCM_EARLY owner_assigned key=%s attempt=%s dp=%s owner=%s candidates=%s "
                 "protected_charge_bytes=%s domain_used_bytes=%s domain_limit_bytes=%s domain_peak_bytes=%s "
                 "io_quota_bytes=%s io_limit_bytes=%s assignments=%s ns=%s "
                 "native_load_headroom_bytes=%s native_dump_headroom_bytes=%s",
                 job["key"], job["attempt"], job["dp"], owner, work, charge, budget["used"],
                 budget["limit"], budget["peak"], budget["io_quota"], budget["io_limit"],
                 load["assignments"], job["assigned_ns"], budget["native_load_headroom_bytes"],
                 budget["native_dump_headroom_bytes"])

    def confirm_batch(self, job, message):
        """Control-thread atomic unique-block grant; never wait on partial claims."""
        index, begin, count = (message.get(k) for k in ("batch", "begin", "count"))
        previous = job["batch_grants"].get(index)
        if previous is not None:
            old_begin, old_count, granted = previous
            if (begin, count) != (old_begin, old_count):
                self.poison("confirmed batch identity changed")
                return
        else:
            valid = (isinstance(index, int) and index == len(job["batch_grants"])
                     and isinstance(begin, int) and begin == job["confirmed"]
                     and isinstance(count, int) and 0 <= count <= 256
                     and begin + count <= len(job["blocks"]))
            if not valid:
                self.poison("invalid confirmed prefix range")
                return
            budget = job["budget"]
            claims = set(job["blocks"][begin:begin + count]) - job["claims"]
            charge = sum(block not in budget["claims"] for block in claims) * budget["block_bytes"]
            granted = (job["status"] == "PENDING" and not job.get("retiring")
                       and budget["used"] + charge <= budget["limit"]
                       and (not count or budget["io_quota"] >= self.capabilities[job["owner"]]["native"]["shard_bytes"]))
            if granted:
                for block in claims:
                    budget["claims"][block] = budget["claims"].get(block, 0) + 1
                job["claims"].update(claims)
                job["charge"] = len(job["claims"]) * budget["block_bytes"]
                job["confirmed"] += count
                budget["used"] += charge
                budget["peak"] = max(budget["peak"], budget["used"])
            job["batch_grants"][index] = (begin, count, granted)
            log.info("UCM_EARLY owner_batch_budget key=%s attempt=%s owner=%s batch=%s "
                     "begin=%s hit=%s granted=%s incremental_bytes=%s domain_used_bytes=%s "
                     "domain_limit_bytes=%s domain_peak_bytes=%s",
                     job["key"], job["attempt"], job["owner"], index, begin, count, granted,
                     charge, budget["used"], budget["limit"], budget["peak"])
        self.send(job["owner"], dict(op="batch_grant", key=job["key"], attempt=job["attempt"],
                                    batch=index, granted=granted))

    def _return_claims(self, job, retained):
        """Caller has a READY-tail or owner-drained proof; control thread only.

        READY means Reserve has discarded its non-prefix groups and every data
        batch completed. Only those tail claims can be returned before drain.
        Overlapping jobs keep independent claims, including cancelled jobs
        whose accepted I/O/native consumers have not drained yet.
        """
        budget = job["budget"]
        released = 0
        for block in job["claims"] - retained:
            count = budget["claims"][block] - 1
            if count:
                budget["claims"][block] = count
            else:
                del budget["claims"][block]
                released += budget["block_bytes"]
        job["claims"] = retained
        job["charge"] = len(retained) * budget["block_bytes"]
        budget["used"] -= released
        return released

    def poison(self, reason):
        if self.fenced:
            return
        self.fenced = True
        self.fence_reason = reason
        # Non-robust native locks/refcounts have no owner-death recovery. Keep
        # credits charged; never authorize a new owner on this shared object.
        for peer in set(self.peers) | self.cohort():
            self.send(peer, dict(op="domain_lost", reason=reason))
        for job in self.jobs.values():
            job["status"] = "FAILED"
        log.error("UCM_EARLY domain_fenced reason=%s credits_retained=true", reason)

    def retire(self, job, reason):
        if job.get("retiring"):
            return
        job["retiring"] = True
        job["status"] = "CANCELLED"
        for peer in self.workers(job["dp"]):
            self.send(peer, dict(op="retire", key=job["key"], attempt=job["attempt"], reason=reason))

    def receive(self, peer, message, now=None):
        op = message["op"]
        if self.fenced and op in ("hello", "job"):
            self.send(peer, dict(op="domain_lost", reason=self.fence_reason))
        if op == "domain_fault" and peer in self.cohort() and peer in self.sessions:
            self.poison(f"{peer}: {message.get('reason', 'unknown drain state')}")
            return
        if self.fenced and op not in ("hello", "ping"):
            return  # Never return credits or release shared references after fencing.
        if op in ("hello", "ping"):
            session = message.get("session")
            if session and peer in self.sessions and session != self.sessions[peer]:
                if self.domains:
                    self.poison("control participant incarnation changed")
            if session:
                self.sessions[peer] = session
            if "capability" in message:
                if self.domains and peer in self.capabilities and message["capability"] != self.capabilities[peer]:
                    self.poison("native capability changed after shared admission")
                self.capabilities[peer] = message["capability"]
            if op == "hello" and self.domains:
                self.send(peer, dict(op="domain_active"))
        if op == "job":
            if peer != f"s{message['dp']}" or not 0 <= message["dp"] < self.dp_size:
                return
            sequence = message.get("sequence")
            if sequence is not None:
                if sequence <= self.last_sequence.get(peer, -1):
                    return
                self.last_sequence[peer] = sequence
        job = self.jobs.get(message.get("key"))
        if job and job.get("owner"):
            self.peers[peer] = time.monotonic() if now is None else now
            if op in ("finish", "cancel", "handoff"):
                if peer != job["client"]:
                    return
                if op in ("finish", "cancel"):
                    self.retire(job, op)
                return  # metadata carries the generation; submission is no acknowledgement
            if peer not in self.workers(job["dp"]) or message.get("attempt") != job["attempt"]:
                return
            if op == "confirm_batch" and peer == job["owner"]:
                self.confirm_batch(job, message)
                return
            if op == "owner_answer" and peer == job["owner"] and job["status"] == "PENDING":
                if message["state"] != "READY" or message.get("hit") != job["confirmed"]:
                    self.send(job["client"], dict(op="result", key=job["key"], state="FAILED", reason=message.get("reason")))
                    self.retire(job, "owner failure")
                else:
                    refunded = self._return_claims(job, set(job["blocks"][:message["hit"]]))
                    job["status"] = "READY"
                    log.info("UCM_EARLY owner_prefix_budget key=%s attempt=%s owner=%s hit=%s "
                             "returned_bytes=%s retained_claim_bytes=%s domain_used_bytes=%s",
                             job["key"], job["attempt"], job["owner"], message["hit"],
                             refunded, job["charge"], job["budget"]["used"])
                    log.info("UCM_EARLY ready key=%s attempt=%s dp=%s owner=%s total_hit_blocks=%s ns=%s",
                             job["key"], job["attempt"], job["dp"], job["owner"], message["hit"], time.monotonic_ns())
                    self.send(job["client"], dict(op="result", key=job["key"], state="READY", hit=message["hit"],
                                                  owner=job["owner"], attempt=job["attempt"], grant=job["grant"]))
            elif op == "consumed" and job["status"] == "READY":
                job["consumers"].add(peer)
                if job["consumers"] == self.workers(job["dp"]):
                    self.retire(job, "all native loads completed")
            elif op == "consumer_failed":
                self.poison("consumer failure did not prove native queue drain")
            elif op == "worker_retire":
                self.send(job["client"], dict(op="result", key=job["key"], state="FAILED", reason=message.get("reason", "worker shutdown")))
                self.retire(job, message.get("reason", "worker shutdown"))
            elif op == "quiesced" and job.get("retiring"):
                job["terminal"].add(peer)
                if job["terminal"] == self.workers(job["dp"]):
                    self.send(job["owner"], dict(op="release_owner", key=job["key"], attempt=job["attempt"]))
            elif op == "owner_drained" and peer == job["owner"] and job.get("retiring") and job["terminal"] == self.workers(job["dp"]):
                budget = job["budget"]
                released = self._return_claims(job, set())
                load = budget["owners"][peer]
                load["jobs"] -= 1
                load["candidates"] -= len(job["blocks"])
                for consumer in self.workers(job["dp"]):
                    self.send(consumer, dict(op="forget", key=job["key"], attempt=job["attempt"]))
                log.info("UCM_EARLY owner_released key=%s attempt=%s dp=%s owner=%s bytes=%s "
                         "domain_used_bytes=%s protection_ms=%.3f completed_read_bytes=%s "
                         "submitted_read_bytes=%s wasted_read_upper_bound_bytes=%s ns=%s",
                         job["key"], job["attempt"], job["dp"], peer, released, budget["used"],
                         (time.monotonic_ns() - job["assigned_ns"]) / 1e6,
                         message.get("read_bytes"), message.get("submitted_read_bytes"),
                         message.get("wasted_read_upper_bound_bytes"), time.monotonic_ns())
                self.jobs.pop(job["key"])
            return
        if op in ("confirm_batch", "owner_answer", "consumed", "quiesced", "owner_drained", "consumer_failed", "worker_retire"):
            return
        if op in ("finish", "cancel", "handoff") and job and job["client"] != peer:
            return
        return super().receive(peer, message, now)

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        if self.domains and any(now - seen > 60 for seen in self.peers.values()):
            self.poison("participant heartbeat lost; death/refcount state unknown")
            return
        for job in list(self.jobs.values()):
            if job.get("owner") and job["status"] == "PENDING" and now - job["created"] > self.timeout:
                self.send(job["client"], dict(op="result", key=job["key"], state="FAILED", reason="owner timeout"))
                self.retire(job, "owner timeout")
        # Owner timeouts above are already retiring; a domain peer loss exits
        # through poison. Reuse all native broker loss/timeout behavior for
        # legacy jobs, including invalidating READY after a required peer loss.
        super().expire(now)
