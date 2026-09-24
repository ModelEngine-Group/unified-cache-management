# 版权所有（c）华为技术有限公司 2012-2026

import threading
from typing import Dict, List, Tuple

class LightweightTokenCache:
    def __init__(self, tokenier, max_entries: int = 500, rollback_tokens: int=15):
        self.tokenier = tokenier
        self.max_entries = max_entries
        self.rollback_tokens = rollback_tokens
        self.cache: Dict[str, List[int]] = {}
        self.lock = threading.Lock()
        print("token cache init success!!!!!!")

    def try_match(self, prompt: str) -> Tuple[bool, List[int], str]:
        with self.lock:
            best_match = ""
            # 1. 寻找最长匹配的前缀
            for prefix in self.cache.keys():
                if prompt.startswith(prefix) and len(prefix) > len(best_match):
                    best_match = prefix
                
            if not best_match:
                return False, [], prompt

            cache_ids = self.cache.get(best_match)

            # 如果完全一致（无增量）
            if len(prompt) == len(best_match):
                return True, cache_ids, ""

            # 2. 决定回退多少个token, 防止边界Token融合切割错误
            rollback = min(self.rollback_tokens, len(cache_ids))
            if rollback == 0:
                return True, cache_ids ,prompt[len(best_match):]

            safe_ids = cache_ids[:-rollback]
            rolled_back_ids = cache_ids[-rollback:]

            # 3. [核心魔法]：仅解码被截断的”尾巴“，不解码头部，完美避开 BOS 对其问题
            try:
                # 必须保留 special_tokens, 防止尾部刚好切在某个特殊的tag上
                tail_str = self.tokenier.decode(
                    rolled_back_ids,
                    skip_special_tokens=False,
                    clean_up_toeknization_spaces=False
                )
            except Exception:
                # 极端异常情况降级处理
                return True, cache_ids, prompt[len(best_match):]

            # 4. 拼接最终需要重新Tokenize的后缀： 被切掉的尾巴字符串 + 真正新增的文本
            suffix = tail_str + prompt[len(best_match):]

            return True, safe_ids, suffix

    def insert(self, prompt:str, token_ids: List[int]):
        # [修改点]：不再提前Decode校验，直接暴力存入
        # 上游传来的prompt是什么样， 就认什么样， 保证O（1）存储且必定成功。
        if not prompt or not token_ids:
            return
        
        with self.lock:
            if prompt not in self.cache:
                # LRU-like 淘汰机制
                if len(self.cache) >= self.max_entries:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[prompt] = token_ids[:]
