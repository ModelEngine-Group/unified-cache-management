import unifiedcache.ucm_sparse.utils as gsa_config

gsa_config.MAX_TOPK_LEN = 64
gsa_config.MIN_TOPK_LEN = 48



print(gsa_config.compute_topk_len(64))


