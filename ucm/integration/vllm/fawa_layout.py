"""Ascend FAWA layout normalization, independent of the vLLM version."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockGeometry:
    logical: int
    physical: int


def ascend_block_geometry(spec) -> BlockGeometry:
    """Old specs expose physical block_size; new specs expose logical size.

    Some old sliding specs already have storage_block_size, equal to block_size.
    Check the relationship, rather than the presence of that property alone.
    """
    nested = getattr(spec, "kv_cache_specs", None)
    if nested:
        geometries = {ascend_block_geometry(member) for member in nested.values()}
        if len(geometries) != 1:
            raise ValueError(f"Inconsistent Ascend group block geometry: {geometries}")
        return geometries.pop()
    size = int(spec.block_size)
    ratio = int(getattr(spec, "compress_ratio", 1))
    physical = int(getattr(spec, "storage_block_size", size))
    if min(size, ratio, physical) <= 0:
        raise ValueError("Ascend block size and compress ratio must be positive.")
    if size == physical * ratio:
        return BlockGeometry(size, physical)
    if size == physical:
        return BlockGeometry(size * ratio, physical)
    raise ValueError(
        f"Unsupported Ascend block geometry: block_size={size}, "
        f"storage_block_size={physical}, compress_ratio={ratio}."
    )


def select_transfer_views(tensors):
    """Replace the A5 (K, scale, full) alias triple with its full-page view.

    Do not change the original runner tensors. Full is only safe for complete
    physical pages: its apparent token rows reinterpret planar K/scale bytes.
    Return whether the selected view requires full-page transfers.
    """
    tensors = tuple(tensors)
    if len(tensors) != 3:
        return tensors, False
    key, scale, full = tensors
    if key.dim() != 4 or scale.dim() != 4 or full.dim() != 4:
        return tensors, False
    if key.data_ptr() != full.data_ptr():
        return tensors, False
    page_bytes = [t[0].numel() * t.element_size() for t in tensors]
    strides = [t.stride(0) * t.element_size() for t in tensors]
    same_storage = (
        len({(str(t.device), t.untyped_storage().data_ptr()) for t in tensors}) == 1
    )
    if not (
        same_storage
        and key.shape[:3] == scale.shape[:3] == full.shape[:3]
        and all(t[0].is_contiguous() for t in tensors)
        and len(set(strides)) == 1
        and strides[0] >= page_bytes[2]
        and scale.data_ptr() == key.data_ptr() + page_bytes[0]
        and page_bytes[0] + page_bytes[1] == page_bytes[2]
    ):
        raise ValueError("Invalid overlapping Ascend K/scale/full cache views.")
    return (full,), True
