"""Execution eligibility on the broker's verified physical L1 domain.

This is a safe-state test, not a second allocator. Only selected, confirmed
batches enter held; native TryGet still decides actual allocation. A holder's
maximum is a conservative completion bound, never an up-front charge.
"""
from dataclasses import dataclass, field


@dataclass
class Claim:
    maximum: set
    held: set = field(default_factory=set)
    completed: bool = False


class ExecutionWindow:
    def __init__(self, blocks):
        self.capacity = blocks
        self.claims = {}

    def add(self, key, maximum):
        self.claims[key] = Claim(set(maximum))

    def narrow(self, key, maximum):
        claim = self.claims[key]
        maximum = set(maximum)
        if not maximum <= claim.maximum:
            raise ValueError("completion bound must not grow")
        claim.maximum = maximum

    def safe(self, key, blocks):
        projected = {k: c.held | (set(blocks) if k == key else set())
                     for k, c in self.claims.items()}
        # Only actual holders participate. Non-holders wait without pins.
        pending = {k for k, held in projected.items() if held}
        while pending:
            live = set().union(*(projected[k] for k in pending))
            if len(live) > self.capacity:
                return False
            finishing = next((k for k in pending
                              if self.claims[k].completed or
                              len(live | self.claims[k].maximum) <= self.capacity), None)
            if finishing is None:
                return False
            pending.remove(finishing)
        return True

    def acquire(self, key, blocks):
        blocks = set(blocks)
        claim = self.claims[key]
        if not blocks <= claim.maximum or not self.safe(key, blocks):
            return False
        claim.held.update(blocks)
        return True

    def complete(self, key):
        self.claims[key].completed = True

    def release(self, key):
        self.claims.pop(key, None)

    @property
    def used(self):
        return len(set().union(*(c.held for c in self.claims.values())))
