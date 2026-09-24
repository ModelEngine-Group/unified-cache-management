# 版权所有（c）华为技术有限公司 2012-2026
import copy
import hashlib
import math
import os 
import pickle
import time
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import numpy as np


@dataclass
class KvbVllmRequestMeta:
    slot_indices: list[bool] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    num_pruned_blocks: int = 0
    ucm_slot_indices: list[bool] = field(default_factory=list)
    vllm_load_slot_indices: list[bool] = field(default_factory=list)
    req_hash: str | None = None


def get_layer_idx(layer_name: str):
    match = re.search(r'(?:layers|blocks|h)\.(\d+)',layer_name)
    if match:
        return int(match.group(1))
    return None
