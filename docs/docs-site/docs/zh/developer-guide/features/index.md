# 特性

特性文档把一项已支持的特性从零带到验证完成：特性是什么、如何启用、如何测量效果。这里不展开内部实现细节，深入设计见[设计文档](../design-docs/index.md)。

| 特性 | 能力 |
| --- | --- |
| [多模态 Encoder Cache 复用](ec-connector.md) | 把 vLLM 的 encoder 缓存存入 UCM 分层存储，相同媒体内容的请求跳过 encoder 重算 |
