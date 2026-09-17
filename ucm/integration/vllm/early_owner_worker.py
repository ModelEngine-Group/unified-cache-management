"""Owner/consumer lifecycle using the existing worker, jobs and data thread."""
import logging
import time

from .early_consumer import Consumer, LoadConsistency

log = logging.getLogger(__name__)
quarantine = []  # Lifetime ends only with the fenced worker process.


class OwnerWorker:
    def __init__(self, worker, capability, peer):
        self.worker, self.capability, self.peer = worker, capability, peer
        self.fatal = None
        self.consumer = Consumer(peer, self.send, self.poison)
        self.release_requested = set()
        self.last_grant = 0
        self.domain_active = False
        if capability.get("ready"):
            worker.store.set_consumer(self.consumer)
            connector = worker.connector
            connector._rank_consistency = LoadConsistency(connector._rank_consistency, self.consumer)

    def send(self, message):
        self.worker.control.send(message)

    def check(self):
        if self.fatal:
            raise RuntimeError("Early shared domain fenced; restart all members with fresh native L1: " + self.fatal)

    def poison(self, reason, announce=True):
        if self.fatal:
            return
        self.fatal = reason
        self.consumer.fenced = reason
        # Quarantine references. Native shared counters and spinlocks do not
        # recover owner death; a heartbeat timeout cannot authorize decrement.
        if announce:
            self.send(dict(op="domain_fault", reason=reason))
        log.error("UCM_EARLY domain_fenced consumer=%s reason=%s recovery=fresh_native_L1", self.peer, reason)

    def receive(self, message):
        op = message["op"]
        if self.fatal:
            return True
        if op == "domain_active":
            self.domain_active = True
            self.consumer.domain_active = True
            return True
        if op == "domain_lost":
            self.poison(message["reason"], announce=False)
            return True
        if op == "lost" and (self.domain_active or any(j.owner for j in self.worker.jobs.values())):
            self.poison(message.get("reason", "control lost"), announce=False)
            return True
        if op in ("owner_reserve", "observe"):
            self.check()
            if not self.capability.get("ready"):
                self.poison("owner grant without verified capability")
                return True
            if message["grant"] <= self.last_grant:
                return True
            self.last_grant = message["grant"]
            from .early_prefetch import WorkerJob
            with self.worker.condition:
                key = message["key"]
                if key in self.worker.jobs:
                    return True
                job = WorkerJob(key, message.get("blocks", []), owner=message["owner"],
                                attempt=message["attempt"], grant=message["grant"],
                                request_id=message.get("request"))
                self.worker.jobs[key] = job
                self.worker.io_limit = message["io_quota"]
                if message.get("request"):
                    self.worker.request_keys[message["request"]] = key
                log.info("UCM_EARLY owner_observed key=%s attempt=%s owner=%s consumer=%s grant=%s ns=%s",
                         key, job.attempt, job.owner, self.peer, job.grant, time.monotonic_ns())
                if op == "owner_reserve":
                    if job.owner != self.peer:
                        self.poison("misrouted owner grant")
                        return True
                    log.info("UCM_EARLY lookup_enqueued key=%s attempt=%s owner=%s rank=%s "
                             "ns=%s submitted_ns=%s", key, job.attempt, job.owner,
                             self.worker.connector.tp_rank, time.monotonic_ns(), message.get("submitted_ns"))
                    self.worker.lookup_pool.submit(self.worker._reserve, job)
                else:
                    job.lookup_finished = True
                self.worker.condition.notify_all()
            return True
        if op == "batch_grant":
            with self.worker.condition:
                job = self.worker.jobs.get(message["key"])
                if job and job.attempt == message["attempt"] and job.owner == self.peer:
                    job.batch_grants[message["batch"]] = message["granted"]
                    self.worker.condition.notify_all()
            return True
        if op not in ("retire", "release_owner", "forget"):
            return False
        key, attempt = message["key"], message["attempt"]
        with self.worker.condition:
            job = self.worker.jobs.get(key)
            if job is None or not job.owner or job.attempt != attempt:
                return True
            if op == "retire":
                job.release_reason = message.get("reason", "terminal")
                self.consumer.retire(key, attempt)
                # Stop prefetch before READY; native consumption cannot have
                # begun. After READY keep every pin until all consumers quiesce.
                if job.owner == self.peer and not job.ready:
                    self.worker._release(job)
            elif op == "release_owner" and job.owner == self.peer:
                self.worker._release(job)
                self.release_requested.add(key)
            elif op == "forget":
                self.consumer.forget(key, attempt)
                self.worker.jobs.pop(key, None)
                if self.worker.request_keys.get(job.request_id) == key:
                    self.worker.request_keys.pop(job.request_id, None)
            self.worker.condition.notify_all()
        return True

    def bind(self, metadata):
        self.check()
        tickets = getattr(metadata, "early_owners", {})
        live, retired = {}, {}
        for request_id, ticket in tickets.items():
            job = self.worker.jobs.get(ticket["key"])
            if job is not None and (job.attempt != ticket["attempt"] or job.owner != ticket["owner"]):
                self.poison("metadata generation differs from owner grant")
                self.check()
            if job is None and ticket["grant"] <= self.last_grant:
                retired[request_id] = (ticket["key"], ticket["attempt"])
            else:
                live[request_id] = ticket
        self.consumer.bind(live, self.capability.get("layers", []))
        # Decode metadata can legitimately repeat an already consumed ticket.
        # It authorizes no new load; begin() rejects the absent generation.
        with self.consumer.lock:
            self.consumer.requests.update(retired)

    def answer(self, job, error):
        if job.cancelled or job.ready:
            return
        if error is not None:
            self.worker._release(job)
        else:
            job.ready = True
        hit = sum(b.reservation["total_hit_blocks"] for b in job.batches)
        self.send(dict(op="owner_answer", key=job.key, attempt=job.attempt,
                       state="FAILED" if error else "READY", reason=error,
                       hit=hit))
        fast = error is None and hit > 0 and not any(b.started for b in job.batches)
        outcome = "failed" if error else "zero_hit" if not hit else "l1_ready" if fast else "prefetched"
        log.info("UCM_EARLY owner_ready key=%s attempt=%s owner=%s rank=%s success=%s hit=%s fast_path=%s outcome=%s ns=%s",
                 job.key, job.attempt, self.peer, self.worker.connector.tp_rank, error is None,
                 hit, fast, outcome, time.monotonic_ns())
        if fast:
            log.info("UCM_EARLY fast_path key=%s attempt=%s rank=%s data_queue_wait=0 l2_read_shards=0",
                     job.key, job.attempt, self.worker.connector.tp_rank)

    def tick(self):
        if self.fatal:
            return
        with self.worker.condition:
            for key in list(self.release_requested):
                job = self.worker.jobs.get(key)
                if job is None or not job.lookup_finished:
                    continue
                if any(b.started and not b.done for b in job.batches):
                    continue  # the existing data thread still owns accepted I/O
                snapshots = [b.reservation["batch"].snapshot() for b in job.batches]
                submitted = sum(s["l2_read_bytes"] for s in snapshots)
                completed = sum(s["completed_read_bytes"] for s in snapshots)
                job.batches.clear()
                self.release_requested.remove(key)
                self.send(dict(op="owner_drained", key=key, attempt=job.attempt,
                               submitted_read_bytes=submitted, read_bytes=completed,
                               wasted_read_upper_bound_bytes=(0 if getattr(job, "release_reason", None)
                                   == "all native loads completed" else completed)))

    def shutdown(self):
        # Normal shutdown attempts the same generation-scoped retire/drain
        # handshake as cancellation. Unknown accelerator completion or a lost
        # participant cannot be converted into a guessed reference decrement.
        with self.worker.condition:
            for job in self.worker.jobs.values():
                if job.owner:
                    self.send(dict(op="worker_retire", key=job.key, attempt=job.attempt))
            config = getattr(self.worker.store, "config", {})
            timeout = config.get("timeout_ms", 30000) / 1000 + 1 if isinstance(config, dict) else 1
            drained = self.worker.condition.wait_for(
                lambda: self.fatal or not any(j.owner for j in self.worker.jobs.values()), timeout=timeout)
        if not drained:
            self.poison("shutdown drain handshake timed out")
