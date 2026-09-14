"""Single owner protocol on the existing control channel and job table.

Credits cover whole candidates before work starts. They are returned only after
an owner drain acknowledgement, never on a timer or a presumed process death.
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
VERSION = 1


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
        # Actual object identity, physical host and namespace reachability.
        domain = [Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  os.readlink("/proc/self/ns/ipc"), os.readlink("/proc/self/ns/mnt"), native["backing"]]
        layout = [connector.unique_id, native, config["tensor_size_list"],
                  connector.use_layerwise, connector.hash_block_size, connector.blocks_per_chunk,
                  connector.generate_hash(connector.hash_block_size,
                                          [0] * connector.hash_block_size, connector._seed)[0].hex()]
        return dict(version=VERSION, domain=json.dumps(domain),
                    layout=hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
                    unique_id=str(connector.unique_id), native=native, layers=layers,
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
            limit = native["reservable_nodes"] * native["shard_bytes"]
            io = min(limit, len(members) * 128 * native["shard_bytes"])
            quota = io // len(members) // native["shard_bytes"] * native["shard_bytes"]
            if quota == 0:
                return None, "insufficient native load-exclusive headroom"
            self.domains[domain] = dict(limit=limit, used=0, peak=0, io_limit=io,
                                        io_quota=quota, members=members, owners={})
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
        cap = self.capabilities[next(iter(self.workers(job["dp"])))]
        work = len(job["blocks"])
        charge = work * cap["native"]["block_bytes"]
        candidates = []
        for peer in self.workers(job["dp"]):
            load = budget["owners"].get(peer, dict(jobs=0, candidates=0))
            peer_cap = self.capabilities[peer]
            work_limit = peer_cap["max_candidates"] or budget["limit"] // cap["native"]["block_bytes"]
            if peer in self.peers and load["jobs"] < peer_cap["max_jobs"] and load["candidates"] + work <= work_limit:
                candidates.append(peer)
        owner = self.select_owner(candidates, job["dp"]) if budget["used"] + charge <= budget["limit"] else None
        if owner is None:
            log.info("UCM_EARLY owner_nospace key=%s dp=%s candidates=%s protected=%s limit=%s wasted_read_bytes=0",
                     job["key"], job["dp"], work, budget["used"], budget["limit"])
            self.fail(job, "bounded owner capacity; native fallback")
            return
        load = budget["owners"].setdefault(owner, dict(jobs=0, candidates=0, assignments=0))
        load["jobs"] += 1
        load["candidates"] += work
        load["assignments"] += 1
        budget["used"] += charge
        budget["peak"] = max(budget["peak"], budget["used"])
        self.next_grant += 1
        job.update(owner=owner, attempt=uuid.uuid4().hex, grant=self.next_grant, running=True, budget=budget,
                   charge=charge, consumers=set(), terminal=set(), assigned_ns=time.monotonic_ns())
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
                 "io_quota_bytes=%s io_limit_bytes=%s assignments=%s ns=%s",
                 job["key"], job["attempt"], job["dp"], owner, work, charge, budget["used"],
                 budget["limit"], budget["peak"], budget["io_quota"], budget["io_limit"],
                 load["assignments"], job["assigned_ns"])

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
            if op == "owner_answer" and peer == job["owner"] and job["status"] == "PENDING":
                if message["state"] != "READY" or not 0 <= message.get("hit", -1) <= len(job["blocks"]):
                    self.send(job["client"], dict(op="result", key=job["key"], state="FAILED", reason=message.get("reason")))
                    self.retire(job, "owner failure")
                else:
                    job["status"] = "READY"
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
                self.send(job["client"], dict(op="result", key=job["key"], state="FAILED", reason="worker shutdown"))
                self.retire(job, "worker shutdown")
            elif op == "quiesced" and job.get("retiring"):
                job["terminal"].add(peer)
                if job["terminal"] == self.workers(job["dp"]):
                    self.send(job["owner"], dict(op="release_owner", key=job["key"], attempt=job["attempt"]))
            elif op == "owner_drained" and peer == job["owner"] and job.get("retiring") and job["terminal"] == self.workers(job["dp"]):
                budget = job["budget"]
                budget["used"] -= job["charge"]
                load = budget["owners"][peer]
                load["jobs"] -= 1
                load["candidates"] -= len(job["blocks"])
                for consumer in self.workers(job["dp"]):
                    self.send(consumer, dict(op="forget", key=job["key"], attempt=job["attempt"]))
                log.info("UCM_EARLY owner_released key=%s attempt=%s dp=%s owner=%s bytes=%s "
                         "domain_used_bytes=%s protection_ms=%.3f completed_read_bytes=%s "
                         "submitted_read_bytes=%s wasted_read_upper_bound_bytes=%s ns=%s",
                         job["key"], job["attempt"], job["dp"], peer, job["charge"], budget["used"],
                         (time.monotonic_ns() - job["assigned_ns"]) / 1e6,
                         message.get("read_bytes"), message.get("submitted_read_bytes"),
                         message.get("wasted_read_upper_bound_bytes"), time.monotonic_ns())
                self.jobs.pop(job["key"])
            return
        if op in ("owner_answer", "consumed", "quiesced", "owner_drained", "consumer_failed", "worker_retire"):
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
