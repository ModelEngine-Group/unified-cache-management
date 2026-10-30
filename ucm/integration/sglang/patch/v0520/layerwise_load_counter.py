"""Compatibility counter for SGLang 0.5.20, which lacks the linker counter."""

from __future__ import annotations

import threading
from concurrent.futures import Future


class LayerWiseLoadCounter:
    """Completion counter compatible with SGLang's KV-pool layer hooks."""

    def __init__(self, num_layers: int, tp_rank: int = 0):
        self.num_layers = num_layers
        self.tp_rank = tp_rank
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        self._lock = threading.Lock()

    def update_producer(self) -> int:
        with self._lock:
            self.producer_index += 1
            index = self.producer_index
            self.futures[index] = [Future() for _ in range(self.num_layers)]
            return index

    def set_consumer(self, index: int) -> None:
        with self._lock:
            self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        with self._lock:
            futures = self.futures.get(index)
            if futures is None:
                raise RuntimeError(f"UCM completion references unknown batch {index}.")
            if not 0 <= layer < len(futures):
                raise IndexError(f"UCM completion has invalid layer {layer}.")
            future = futures[layer]
        if not future.done():
            future.set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        with self._lock:
            futures = tuple(self.futures.get(index, ()))
        for future in futures:
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        with self._lock:
            futures = self.futures.get(self.consumer_index)
        if futures is None:
            return
        if not 0 <= threshold < len(futures):
            raise IndexError(f"UCM wait has invalid layer {threshold}.")
        futures[threshold].result()

    def finish(self, index: int) -> bool:
        with self._lock:
            futures = self.futures.get(index)
        if futures is None:
            return True
        landed = True
        for future in futures:
            try:
                future.result()
            except BaseException:
                landed = False
        with self._lock:
            self.futures.pop(index, None)
        return landed

    def reset(self) -> None:
        with self._lock:
            self.producer_index = -1
            self.consumer_index = -1
            self.futures.clear()
