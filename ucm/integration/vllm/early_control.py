"""UCM control messages for unscheduled jobs and protected READY results.

The endpoint follows the native UCM shared identity. It owns no cache data,
pins, leases, or L1 identities. Each socket is confined to one control thread.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time

import msgspec
import zmq

log = logging.getLogger(__name__)


class BrokerState:
    """Single-threaded aggregation; also exercised without transport in tests."""

    def __init__(self, send, tp_size, timeout=300):
        self.send = send
        self.tp_size = tp_size
        self.timeout = timeout
        self.peers = {}
        self.jobs = {}

    def workers(self, dp):
        return {f"w{dp}.{rank}" for rank in range(self.tp_size)}

    def release(self, job, reason="terminal"):
        for worker in self.workers(job["dp"]):
            self.send(worker, {"op": "release", "key": job["key"], "reason": reason})

    def fail(self, job, reason):
        if job["status"] in ("FAILED", "CANCELLED"):
            return
        job["status"] = "FAILED"
        self.release(job, "failed")
        log.warning("UCM_EARLY failed key=%s reason=%s", job["key"], reason)
        self.send(
            job["client"],
            {"op": "result", "key": job["key"], "state": "FAILED", "reason": reason},
        )

    def dispatch(self, job):
        if job["status"] != "PENDING" or job.get("running"):
            return
        if not self.workers(job["dp"]).issubset(self.peers):
            return
        job["answers"] = {}
        job["running"] = True
        for worker in self.workers(job["dp"]):
            self.send(
                worker,
                {
                    "op": "reserve",
                    "key": job["key"],
                    "blocks": job["blocks"],
                    "request": job.get("request"),
                    "submitted_ns": job.get("submitted_ns"),
                },
            )

    def wake(self):
        for job in self.jobs.values():
            self.dispatch(job)

    def receive(self, peer, message, now=None):
        now = time.monotonic() if now is None else now
        self.peers[peer] = now
        op = message["op"]
        if op in ("hello", "ping"):
            self.send(peer, {"op": "pong"})
            self.wake()
        elif op == "job":
            key = message["key"]
            if key in self.jobs:
                return
            job = dict(message, client=peer, status="PENDING", created=now)
            self.jobs[key] = job
            self.dispatch(job)
        elif op == "answer":
            job = self.jobs.get(message["key"])
            if job is None or job["status"] != "PENDING" or not job.get("running"):
                return
            if peer not in self.workers(job["dp"]):
                return
            if message["state"] != "READY":
                self.fail(job, message.get("reason", "worker failed"))
                return
            job["answers"][peer] = message["hit"]
            if self.workers(job["dp"]) == job["answers"].keys():
                job["status"] = "READY"
                hit = min(job["answers"].values())
                log.info(
                    "UCM_EARLY ready key=%s dp=%s ranks=%s total_hit_blocks=%s ns=%s",
                    job["key"],
                    job["dp"],
                    len(job["answers"]),
                    hit,
                    time.monotonic_ns(),
                )
                self.send(
                    job["client"],
                    {"op": "result", "key": job["key"], "state": "READY", "hit": hit},
                )
        elif op == "handoff":
            job = self.jobs.get(message["key"])
            if job is not None and job["client"] == peer:
                for worker in self.workers(job["dp"]):
                    self.send(worker, message)
        elif op in ("finish", "cancel"):
            job = self.jobs.pop(message["key"], None)
            if job:
                job["status"] = "CANCELLED"
                self.release(job)

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        dead = {peer for peer, seen in self.peers.items() if now - seen > 60}
        for peer in dead:
            self.peers.pop(peer, None)
        for job in self.jobs.values():
            if job["status"] == "PENDING" and now - job["created"] > self.timeout:
                self.fail(job, "bounded pending timeout")
            elif dead.intersection(self.workers(job["dp"]) | {job["client"]}):
                self.fail(job, "control peer lost")


class Control:
    def __init__(self, endpoint, identity, callback, broker_config=None, hello=None):
        self.endpoint, self.identity, self.callback = endpoint, identity, callback
        self.broker_config = broker_config
        import uuid
        self.hello = dict(hello or {}, session=uuid.uuid4().hex)
        self.outbound = queue.SimpleQueue()
        self.reader, self.writer = socket.socketpair()
        self.writer.setblocking(False)
        self.stopping = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="ucm-early-control", daemon=True
        )
        self.thread.start()

    def send(self, message):
        if self.stopping.is_set():
            return
        self.outbound.put(message)
        try:
            self.writer.send(b"x")
        except BlockingIOError:
            pass  # already readable; the queue is drained completely

    def close(self):
        self.stopping.set()
        try:
            self.writer.send(b"x")
        except (OSError, BlockingIOError):
            pass
        self.thread.join(timeout=5)

    def _run(self):
        context = zmq.Context()
        dealer = context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.IDENTITY, self.identity.encode())
        dealer.setsockopt(zmq.LINGER, 0)
        router = None
        encode, decode = msgspec.msgpack.encode, msgspec.msgpack.decode
        try:
            poller = zmq.Poller()
            if self.broker_config is not None:
                router = context.socket(zmq.ROUTER)
                router.setsockopt(zmq.LINGER, 0)
                router.bind(self.endpoint)

                def route(peer, message):
                    router.send_multipart([peer.encode(), encode(message)])

                from .early_owner import OwnerBroker
                broker = OwnerBroker(route, **self.broker_config)
                poller.register(router, zmq.POLLIN)
            dealer.connect(self.endpoint)
            poller.register(dealer, zmq.POLLIN)
            poller.register(self.reader.fileno(), zmq.POLLIN)
            dealer.send(encode(dict(self.hello, op="hello")))
            heartbeat, last_reply = 0.0, None
            connecting_since = time.monotonic()
            connect_timeout = float(os.getenv("VLLM_ENGINE_READY_TIMEOUT_S", "3600"))
            while not self.stopping.is_set():
                events = dict(
                    poller.poll(1000)
                )  # control heartbeat; no storage polling
                if self.reader.fileno() in events:
                    self.reader.recv(65536)
                    while not self.outbound.empty():
                        dealer.send(encode(self.outbound.get()))
                if router is not None and router in events:
                    peer, payload = router.recv_multipart()
                    broker.receive(peer.decode(), decode(payload))
                if dealer in events:
                    message = decode(dealer.recv())
                    last_reply = time.monotonic()
                    if message["op"] != "pong":
                        self.callback(message)
                now = time.monotonic()
                if now - heartbeat >= 2:
                    dealer.send(encode({"op": "ping", "session": self.hello["session"]}))
                    heartbeat = now
                    if router is not None:
                        broker.expire(now)
                if (
                    last_reply is None and now - connecting_since > connect_timeout
                ) or (last_reply is not None and now - last_reply > 60):
                    raise TimeoutError("Early control broker unavailable")
        except Exception as error:
            log.exception("UCM_EARLY control failed identity=%s", self.identity)
            self.callback({"op": "lost", "reason": str(error)})
        finally:
            self.stopping.set()
            dealer.close()
            if router is not None:
                router.close()
            context.term()
            self.reader.close()
            self.writer.close()
