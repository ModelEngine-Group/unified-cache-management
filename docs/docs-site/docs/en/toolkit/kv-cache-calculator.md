---
template: calculator.html
---

# KV Cache Size Calculator

The KV Cache calculator estimates KV memory usage from model architecture, token count, batch size and parallelism settings. It runs in the browser without starting a model, helping you plan capacity for context length and concurrent requests before deployment.

Use it when:

- Estimating how much memory to reserve for KV Cache before increasing context length or batch size.
- Comparing per-device and cluster KV capacity under the displayed formulas when changing KV data types or TP/DP settings.
- Estimating total token capacity from a known memory budget available to KV Cache.

Choose a preset model or enter a supported model URL, set the execution parameters, and inspect the capacity result and formula. For the reverse token calculation, enter memory available to KV after subtracting model weights, runtime overhead and reserved space. Results follow the model formulas shown on the page; verify the inference engine's actual KV layout and TP sharding or replication behavior before deployment.

## Start calculating
