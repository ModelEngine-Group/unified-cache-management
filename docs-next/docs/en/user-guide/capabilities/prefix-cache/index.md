# Prefix Cache

## Prefix Cache: A Fundamental Acceleration Component for KVCache and Its Architectural Considerations in Large Language Model Inference

As the simplest and most fundamental acceleration feature of KVCache, Prefix Cache has achieved industry-wide consensus.
With the expanding application scope of large language models (LLMs), the growth of sequence lengths, and the
proliferation of Agent-based applications, the performance gains of Prefix Cache become even more pronounced.

The core performance metric of Prefix Cache is the hit rate, and there exists a direct positive correlation between
cache capacity and hit rate. Taking the publicly released data from DeepSeek and Kimi as examples, a relatively large
cache capacity is required to reach the "hit rate sweet spot". In terms of input/output (IO) characteristics, Prefix
Cache primarily demands bandwidth-intensive IO, making it well-suited for storage on Solid-State Drives (SSDs).

Prefix Cache can leverage diverse storage media, including Dynamic Random-Access Memory (DRAM), SSDs, and dedicated
storage systems (e.g., DeepSeek's 3fs, a storage system specifically developed for KVCache). The fundamental design
philosophy involves constructing a **multi-level cache** hierarchy using DRAM, local SSDs, and remote storage.

In practice, the implementation of this hierarchy can be roughly categorized into two architectural directions:

- **Decentralized Architecture**: KVCache is deployed in an isolated manner for each inference instance, with each
  KVCache partition belonging to a distinct inference instance (or server). This distributed KVCache deployment is
  typically paired with upper-layer KVCache-aware affinity scheduling. The goal of such scheduling is to route
  inference requests to instances with higher KVCache hit rates, thereby maximizing overall system performance.

- **Centralized Architecture**: KVCache is stored in a centralized external storage system and shared across all
  computing nodes. This architecture features inherent simplicity; DeepSeek's 3fs adopts this design paradigm, and
  the Prefix Cache module in UCM also tends to prioritize this centralized approach.

## Storage Backends

UCM supports multiple storage backends for Prefix Cache:

- **Pipeline Store**: In-memory storage using pipeline architecture
- **NFS Store**: Network File System based storage
- **DS3FS Store**: DeepSeek's 3FS storage system
- **Mooncake Store**: Mooncake-based storage backend
- **Compress Store**: Compressed storage for reduced footprint

Each backend has different performance characteristics and is suitable for different deployment scenarios. Choose the
appropriate backend based on your performance requirements, available infrastructure, and cost considerations.