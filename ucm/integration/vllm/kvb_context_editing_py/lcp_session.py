# 版权所有（c）华为技术有限公司 2012-2026
"""LCP-based session prefix matching for codeagent multi-round requests.

Replaces the old fixed-window token hash (``[8000:10000]``) approach used to 
identify "the same session" across multiple rounds of a codeagent request.

Each request is matched to the previous round of the same session by computing
the longest common prefix (LCP) of the registered token sequences in a shared
in-process registry. A deterministic session key is derived and shared by both
the UCM connector (``kvb_agent_connector``) and the model runner
(``model_runner_v1``), so sparse KV metadata computed in round N can be reused
in round N+1 of the same session without a fragile fixed token window.

The registry lives at module level because the connector and the runner run in
the same worker process and must agree on the same key for the same physical
request. It is single-threaded in the inference hot path and guarded by a lock
for safety.
"""
import hashlib
import threading
from array import array

# Minimum common-prefix length (in tokens) required to treat two requests as
# belonging to the same session. Tuned to the old CAL_HASH window behaviour.
MIN_LCP_TO_MATCH = 10000

#Upper bound on the number of concurrently tracked sessions to avoid
# unbounded memory growth
MAX_SESSIONS = 400

_session_token_map: dict[str, list[int]] = {}
_lock = threading.Lock()


def _compute_lcp(prev: list[int], cur: list[int]) -> int:
    for i, (p, c) in enumerate(zip(prev, cur)):
        if p != c:
            return i
    return min(len(prev), len(cur))


def _derive_new_key(tokens: list[int]) -> str:
    data = array("I",tokens).tobytes()
    return hashlib.sha256(data).hexdigest()


def match_session_key(all_token_ids) -> str:
    """Return the deterministic session key for ``all_token_ids``.

    If the request shares at least ``MIN_LCP_TO_MATCH`` prefix tokens with a 
    previously registered session, the existing session key is returned and the 
    registered token sequence is refreshed to the current round. Otherwise a 
    brand-new session key is derived from the full token sequence.

    The returned key is stable across rounds of the same session, allowing the 
    sparse KV metadata computed in an earlier round to be looked up and reused.
    """
    tokens = list(all_token_ids)
    with _lock:
        best_key = ""
        best_lcp = 0
        if len(tokens) >= MIN_LCP_TO_MATCH:
            for key, prev in _session_token_map.items():
                if len(prev) < MIN_LCP_TO_MATCH:
                    continue
                lcp = _compute_lcp(prev, tokens)
                if lcp > best_lcp:
                    best_lcp, best_key = lcp, key
        else:
            return ""
        if best_key is not None and best_lcp >= MIN_LCP_TO_MATCH:
            _session_token_map[best_key] = tokens
            return best_key
        
        new_key = _derive_new_key(tokens)
        if len(_session_token_map) >= MAX_SESSIONS:
            _session_token_map.clear()
        _session_token_map[new_key] = tokens
        return new_key


def get_request_hash(tokens: list[int], response_index: int):
    data = array("I", tokens[:response_index]).tobytes()
    return hashlib.sha256(data).hexdigest()

