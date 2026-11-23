"""Registered device pages used to write the valid prefix of a KV block."""

import numpy as np
import torch


class PartialTailBuffer:
    """Keep one scratch page per KV component, in the source page's byte layout.

    Each layer can be staged independently. The caller must finish the previous
    transfer from a layer before staging it again.
    """

    def __init__(self, caches, layout, block_size):
        # Match the Store's registered segments, including K/V-first tensors,
        # layerwise ghost pointers and segments cut out of the same page.
        components = {}
        segment_bases = set(map(int, layout.base_ptrs.flat))
        for name, entry in caches.items():
            tensors = entry if isinstance(entry, (tuple, list)) else (entry,)
            for tensor in tensors:
                if (
                    tensor.dim() == 5
                    and tensor.shape[0] == 2
                    and not isinstance(entry, (tuple, list))
                    and tensor[1].data_ptr() in segment_bases
                ):
                    pages = tensor.unbind(0)
                else:
                    pages = (tensor,)
                for source in pages:
                    component = self._make_component(source, name, block_size)
                    components[source.data_ptr()] = component

        self.addresses = np.zeros_like(layout.base_ptrs)
        self.layers = {}
        for index in np.ndindex(self.addresses.shape):
            addr = int(layout.base_ptrs[index])
            if not addr:
                continue
            size = int(layout.tensor_size_lists[index])
            stride = int(layout.block_stride_lists[index])
            component = components.get(addr)
            if component is None:
                # Shared Indexer layouts can split a page across two segments.
                component = next(
                    (
                        item
                        for base, item in components.items()
                        if base < addr < base + item[1].nbytes
                    ),
                    None,
                )
            if component is None:
                raise ValueError("unmapped KV segment")
            source, scratch, _ = component
            offset = addr - source.data_ptr()
            if (
                offset + size > scratch.nbytes
                or stride != source.stride(0) * source.element_size()
            ):
                raise ValueError("KV segment spans more than one physical page")
            self.addresses[index] = scratch.data_ptr() + offset
            row = index[0] if layout.use_layerwise else None
            self.layers.setdefault(row, {})[source.data_ptr()] = component
        self.layers[None] = components
        self.registrations = [
            (scratch.data_ptr(), scratch.nbytes)
            for _, scratch, _ in components.values()
        ]

    @staticmethod
    def _make_component(source: torch.Tensor, name: str, block_size: int):
        page = source[0]
        # Separate K/V pages: [tokens, dim], [tokens, heads, dim] or
        # [heads, tokens, dim]. Combined pages add a leading K/V dimension.
        # Packed byte caches may have scales stored outside the token rows.
        if (
            page.ndim not in (2, 3, 4)
            or not source.is_floating_point()
            or source.element_size() < 2
        ):
            raise ValueError(f"{name}: unsupported packed KV page {tuple(page.shape)}")
        first = 0
        if page.ndim == 4:
            if page.shape[0] != 2:
                raise ValueError(
                    f"{name}: unsupported combined KV page {tuple(page.shape)}"
                )
            first = 1
        axes = [i for i in range(first, page.ndim - 1) if page.shape[i] == block_size]
        if len(axes) != 1:
            raise ValueError(
                f"{name}: no unambiguous token axis in {tuple(page.shape)}"
            )
        scratch = torch.empty_like(page, memory_format=torch.preserve_format)
        if scratch.stride() != page.stride():
            raise ValueError(f"{name}: KV page has inner padding")
        return source, scratch, axes[0]

    def stage(self, block_id, valid_tokens, layer=None):
        """Copy only valid token positions, zeroing all other output bytes."""
        for source, scratch, axis in self.layers[layer].values():
            scratch.zero_()
            scratch.narrow(axis, 0, valid_tokens).copy_(
                source[block_id].narrow(axis, 0, valid_tokens)
            )
        addresses = self.addresses if layer is None else self.addresses[layer]
        return addresses.reshape(1, -1)
