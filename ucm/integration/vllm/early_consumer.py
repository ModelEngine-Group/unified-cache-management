"""Track actual native task completion; metadata/submission never releases pins."""
from contextvars import ContextVar
import logging
import os
import threading
import time

log = logging.getLogger(__name__)
load_scope = ContextVar("ucm_owner_load_scope", default=())


class LoadConsistency:
    """Keep the original consistency object, including all its task state."""
    def __init__(self, native, observer):
        self.native, self.observer = native, observer

    def __getattr__(self, name):
        return getattr(self.native, name)

    def submit_load(self, store, block_ids_by_request, *args, **kwargs):
        token = load_scope.set(self.observer.scope(block_ids_by_request))
        try:
            return self.native.submit_load(store, block_ids_by_request, *args, **kwargs)
        finally:
            load_scope.reset(token)


class Consumer:
    def __init__(self, peer, send, poison):
        self.peer, self.send, self.poison = peer, send, poison
        self.lock = threading.RLock()
        self.generations, self.requests = {}, {}
        self.tasks = {}
        self.serial = 0
        self.fenced = None
        self.max_task_shards = None
        self.max_dump_task_shards = None

    def bind(self, tickets, layers):
        with self.lock:
            self.requests = {}
            for request_id, ticket in tickets.items():
                identity = (ticket["key"], ticket["attempt"])
                self.requests[request_id] = identity
                state = self.generations.setdefault(identity, dict(ticket=ticket, expected=None,
                    plan=None, remaining=None, layers=set(layers), completed={}, active=0,
                    stopped=False, consumed=False, quiesced=False))
                plan = tuple(ticket["expected"])
                if state["expected"] is None:
                    state["plan"] = plan
                    state["expected"] = frozenset(plan)
                    state["remaining"] = len(state["expected"]) * len(state["layers"])
                elif state["plan"] != plan and state["expected"] != set(plan):
                    self.poison("generation changed its native consumption plan")
                    raise RuntimeError("owner consumption plan changed")
                self._notify(state)

    def scope(self, request_ids):
        with self.lock:
            return tuple(self.requests[r] for r in request_ids if r in self.requests)

    def begin(self, ids, indexes):
        with self.lock:
            self.check_task("load", len(indexes))
            identities = load_scope.get()
            if not identities:
                return None
            for identity in identities:
                state = self.generations.get(identity)
                if state is None or state["stopped"] or state["consumed"]:
                    raise RuntimeError("late native load for a retired owner generation")
            raw = memoryview(ids).tobytes()
            shards = list(indexes)
            if len(raw) != len(shards) * 16:
                raise ValueError("native block identity width mismatch")
            ranges = {}
            for i, shard in enumerate(shards):
                ranges.setdefault(int(shard), set()).add(raw[16*i:16*(i+1)])
            self.serial += 1
            token = self.serial
            self.tasks[token] = (identities, ranges, time.monotonic_ns())
            for identity in identities:
                self.generations[identity]["active"] += 1
            return token

    def check_task(self, operation, shards):
        with self.lock:
            if self.fenced:
                raise RuntimeError("native I/O denied on fenced shared domain: " + self.fenced)
            limit = self.max_task_shards if operation == "load" else self.max_dump_task_shards
            if limit is not None and shards > limit:
                self.fenced = f"native {operation} exceeded the advertised allocation headroom bound"
                self.poison(self.fenced)
                raise RuntimeError(self.fenced)

    def unscoped_failure(self):
        with self.lock:
            # A fallback load or native dump shares the same non-robust domain.
            # Its failure matters even when it has no owner-generation receipt.
            if self.max_task_shards is not None and not self.fenced:
                self.fenced = "unscoped native I/O failed; drain state is not proven"
                self.poison(self.fenced)

    def complete(self, token, task, success):
        if token is None:
            return
        with self.lock:
            receipt = self.tasks.pop(token, None)
            if receipt is None:
                return
            identities, ranges, submitted = receipt
            for identity in identities:
                state = self.generations[identity]
                state["active"] -= 1
                if success:
                    for shard, blocks in ranges.items():
                        if shard not in state["layers"]:
                            continue
                        done = state["completed"].setdefault(shard, set())
                        # A native task may batch several requests. Count only
                        # this generation's newly completed expected range;
                        # duplicate chunks/receipts cannot retire later layers.
                        new = blocks.intersection(state["expected"]) - done
                        done.update(new)
                        state["remaining"] -= len(new)
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("UCM_EARLY native_load_completed key=%s attempt=%s consumer=%s task=%s "
                             "layers=%s block_count=%s success=%s submitted_ns=%s completed_ns=%s "
                             "reference_acquired_ns=unknown pid=%s",
                             *identity, self.peer, task, sorted(ranges), sum(map(len, ranges.values())),
                             success, submitted, time.monotonic_ns(), os.getpid())
                if success:
                    self._notify(state)
            if not success:
                # A native failed submission/Wait does not prove all its queue
                # references and accelerator work drained. Quarantine the domain.
                self.fenced = "native consumer load failed; drain state is not proven"
                self.poison(self.fenced)

    def submitted(self, token, task):
        with self.lock:
            identities, ranges, started = self.tasks[token]
            for key, attempt in identities:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("UCM_EARLY native_load_submitted key=%s attempt=%s consumer=%s task=%s "
                             "layers=%s block_count=%s start_ns=%s ns=%s pid=%s",
                             key, attempt, self.peer, task, sorted(ranges), sum(map(len, ranges.values())),
                             started, time.monotonic_ns(), os.getpid())

    def _notify(self, state):
        if self.fenced:
            return
        ticket = state["ticket"]
        envelope = dict(key=ticket["key"], attempt=ticket["attempt"], consumer=self.peer)
        # Completion is maintained by actual successful native Wait receipts.
        # Do not rescan the entire prefix across all prior layers on each Wait.
        complete = state["remaining"] == 0
        if not state["stopped"] and not state["consumed"] and complete and not state["active"]:
            state["consumed"] = True
            self.send(dict(envelope, op="consumed", ns=time.monotonic_ns()))
            log.info("UCM_EARLY native_consumed key=%s attempt=%s consumer=%s ns=%s",
                     ticket["key"], ticket["attempt"], self.peer, time.monotonic_ns())
        if state["stopped"] and not state["active"] and not state["quiesced"]:
            state["quiesced"] = True
            self.send(dict(envelope, op="quiesced", ns=time.monotonic_ns()))
            log.info("UCM_EARLY native_quiesced key=%s attempt=%s consumer=%s ns=%s",
                     ticket["key"], ticket["attempt"], self.peer, time.monotonic_ns())

    def retire(self, key, attempt):
        with self.lock:
            state = self.generations.setdefault((key, attempt), dict(
                ticket=dict(key=key, attempt=attempt), expected=None, layers=set(),
                plan=None, remaining=None, completed={}, active=0, stopped=False,
                consumed=False, quiesced=False))
            state["stopped"] = True
            self._notify(state)

    def forget(self, key, attempt):
        with self.lock:
            state = self.generations.get((key, attempt))
            if state and state["quiesced"] and not state["active"]:
                # Worker grant watermark rejects retired metadata before bind.
                del self.generations[(key, attempt)]
                self.requests = {r: identity for r, identity in self.requests.items()
                                 if identity != (key, attempt)}
