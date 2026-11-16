"""SGLang Store v2 design scaffold, independent of the V1 integration.

linker.py implements initialization and lookup for UcmDirectLinker.
runtime.py initializes one assembler and forwards Scheduler/Worker calls.
device_layout.py defines the DSA, DeepSeek V4 and Mamba object layouts and
their block identities.
"""
