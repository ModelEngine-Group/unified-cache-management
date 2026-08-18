# Sparse Attention Principles

Design principles behind UCM's sparse attention features (GSA and CacheBlend) — the algorithm motivation currently split across `user-guide/capabilities/sparse-attention/index.md`, `gsa.md`, and `cacheblend.md`. The User Guide keeps only the quick-start / enable how-to.

!!! info "Under construction"

    This page consolidates the algorithm rationale moved from the sparse-attention
    User Guide pages. The outline below describes what to write.

## Planned content

- **Why sparse attention**: attention compute as the long-sequence bottleneck; observed sparsity and dispersion in LLM attention (with the existing figures).
- **GSA (Hash-Aware Top-k Attention)**: trainable hash-based similarity; why hash over exact QK; layer-wise adaptive sparsity; query-aware dynamic retrieval; hardware-efficient (CUDA/NPU) kernel design; accuracy and speedup results.
- **CacheBlend**: cached knowledge fusion for RAG; selective recomputation of a small token fraction; the Loading Controller / KV Cache Store / Cache Fusor component model; chunk hash encoding and the shared prefix+chunk hash space; delta-rope postprocess.
- **Interaction with Prefix Cache**: the requirement that sparse algorithms support Prefix Cache; how blend reduces input tokens across all computation kernels, not just attention.
- **Pointers**: link to the GSA/CacheBlend quick-start in User Guide and to the papers.

## Do not

- Do not duplicate quick-start commands or config (User Guide).
- Keep paper links as references; this page is the design rationale, not a paper reprint.

## Acceptance

- A reader can explain each algorithm's mechanism and why it fits UCM.
- Paper links and User Guide quick-start links are present.

Owner: ____
