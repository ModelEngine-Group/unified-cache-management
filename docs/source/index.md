# Welcome to Unified Cache Manager

:::{figure} ./logos/UCM-light.png
:align: center
:alt: UCM
:class: no-scaled-link
:width: 50%
:::

:::{raw} html
<p style="text-align:center">
<strong>Unified Cache Manager
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/ModelEngine-Group/unified-cache-management" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/ModelEngine-Group/unified-cache-management/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/ModelEngine-Group/unified-cache-management/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

The core principle of Unified Cache Manager (UCM) is to persist the LLM KVCache and replace redundant computations
through multiple retrieval mechanisms. UCM not only supports prefix caching but also offers a variety of training-free
sparse attention retrieval methods, delivering higher performance when handling extremely long sequence inference tasks.
Additionally, UCM provides a PD disaggregation solution based on a storage-compute separation architecture, which
enables more straightforward and flexible management of heterogeneous computing resources. When integrated with vLLM,
UCM achieves a 3-10x reduction in inference latency across various scenarios, including multi-turn dialogue and
long-context reasoning tasks.

For more information, check out the following:

* [UCM Section of the ModelEngine Community](https://modelengine-ai.net/#/ucm)

Paper list:

* [HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference](https://arxiv.org/abs/2506.02572)
* [ReTaKe: Reducing Temporal and Knowledge Redundancy for Long Video Understanding](https://arxiv.org/abs/2412.20504)
* [AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding](https://arxiv.org/abs/2503.12559)
* [Dynamic Early Exit in Reasoning Models](https://arxiv.org/abs/2504.15895)
* [Sparse Attention across Multiple-context KV Cache](https://arxiv.org/abs/2508.11661)

## Documentation

:::{toctree}
:caption: Getting Started
:maxdepth: 1
getting-started/quick_start
getting-started/installation_gpu
getting-started/installation_npu
:::

:::{toctree}
:caption: User Guide
:maxdepth: 1
user-guide/prefix-cache/index
user-guide/sparse-attention/index
user-guide/pd-disaggregation/index
:::

:::{toctree}
:caption: Developer Guide
:maxdepth: 1
developer-guide/contribute
:::

:::{toctree}
:caption: About Us
:maxdepth: 1
about
:::
