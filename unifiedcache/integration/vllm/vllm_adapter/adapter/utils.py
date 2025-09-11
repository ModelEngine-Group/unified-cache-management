def patch_dataclass_fields(target_cls, src_cls, *, include_repr=True, include_eq=True, include_hash=True):
    """
    Monkey-patch dataclass field structure from src_cls to target_cls.
    """
    # Core dataclass structure
    target_cls.__init__ = src_cls.__init__
    target_cls.__dataclass_fields__ = src_cls.__dataclass_fields__
    target_cls.__dataclass_params__ = src_cls.__dataclass_params__
    target_cls.__annotations__ = src_cls.__annotations__

    # Optional: additional generated methods
    if include_repr and hasattr(src_cls, "__repr__"):
        target_cls.__repr__ = src_cls.__repr__
    if include_eq and hasattr(src_cls, "__eq__"):
        target_cls.__eq__ = src_cls.__eq__
    if include_hash and hasattr(src_cls, "__hash__"):
        target_cls.__hash__ = src_cls.__hash__