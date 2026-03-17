<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/docs/source/logos/UCM-dark.png">
    <img alt="UCM" src="https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/docs/source/logos/UCM-light.png" width=50%>
  </picture>
</p>

<p align="center">
| <a href="https://ucm.readthedocs.io/en/latest"><b>Documentation</b></a> | <a href="https://modelengine-ai.net/#/ucm"><b>Website</b></a> | <a href="https://github.com/ModelEngine-Group/unified-cache-management/issues/679"><b>RoadMap</b></a> | <a href="README_zh.md"><b>中文</b></a> |
</p>

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/ModelEngine-Group/unified-cache-management)

</div>

---

## Overview

The core principle of Unified Cache Manager (UCM) is to persist the LLM KVCache and replace redundant computations
through multiple retrieval mechanisms. UCM not only supports prefix caching but also offers a variety of training-free
sparse attention retrieval methods, delivering higher performance when handling extremely long sequence inference tasks.
Additionally, UCM provides a PD disaggregation solution based on a storage-compute separation architecture, which
enables more straightforward and flexible management of heterogeneous computing resources. When integrated with vLLM,
UCM achieves a 3-10x reduction in inference latency across various scenarios, including multi-turn dialogue and
long-context reasoning tasks.

### Motivation

With the increase of model size, the KV cache became larger and sparser, especially for long sequence requests. To
reduce the GPU memory used, offload full KV to external storage and only keep partial or compressed KV in GPU memory
became the popular direction. This can also reduce the GPU calculation, increase the sequence length and batch size of
decoding.

Sparse KV cache have many different choices. Recently paper point out that there is no common way can fit all scenarios
and all models. So better to build a common framework then different sparse algorithms can be plugin to it like KV
connector for PC.

![architecture.png](https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/docs/source/_static/images/idea.png)

All gray boxes in the diagram represent existing classes in vLLM version 0.9.2, while the green boxes indicate newly added components by UCM. 
The light green boxes demonstrate potential future subclass extensions based on this framework.

UcmSparseBase is the base class of different sparse algorithms. Just like KV connector design, it will hook few places of
scheduler and layer.py to do additional load, dump and calculate sparse KV blocks.

SparseKVManager allows users to define custom KV block allocations for different algorithms. 
To keep all implementations unified under the SparseKVBase framework, the system calls the SparseKVBase base class, 
while the actual implementation occurs in subclasses of sparse algorithms.

KVStoreBase helps decouple sparse algorithms from external storage. It defines methods for communicating with external storage, 
enabling any sparse algorithm to work seamlessly with any external storage system. 
The core concept here involves identifying blocks through IDs and offsets. 
This approach is not only suitable for sparse scenarios but also naturally accommodates prefix caching. 
The KVStoreConnector links it with the current KVConnectorBase_V1 to provide PC (Prefix Caching) functionality. 
For example, NFSStore serves as a reference implementation that provides the capability to store KVCache 
in either a local filesystem for single-machine scenarios or through NFS mount points in multi-server environments.

---

## Support Features

- Prefix Cache
- Cache Blend
- Model Window Extrapolation
- Prefill Offload
- Sparse Attention
- Sparse Attention Offload
- Heterogeneous PD Disaggregation

---

## Quick Start

please refer to [Quick Start for vLLM](https://ucm.readthedocs.io/en/latest/getting-started/quickstart_vllm.html) and [Quick Start for vLLM-Ascend](https://ucm.readthedocs.io/en/latest/getting-started/quickstart_vllm_ascend.html).

---

## Branch

| **Branch** |     Status | vLLM version |
|-----------:|-----------:|-------------:|
|       main | Maintained |       v0.11.0 |
|    develop | Maintained |       v0.11.0 |

---

## Contact Us
1. For technical questions and feature requests, please use GitHub [Issues](https://github.com/ModelEngine-Group/unified-cache-management/issues).
2. WeChat technical discussion group: Scan the QR code below.

<img src="https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/docs/source/_static/images/qrcode_for_wechat.png" alt="wechat-gh" width="40%">

## License

UCM is licensed under the MIT with additional conditions. Please read the [LICENSE](https://raw.githubusercontent.com/ModelEngine-Group/unified-cache-management/main/LICENSE) file for details.
