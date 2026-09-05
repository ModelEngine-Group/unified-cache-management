# kv-calc — KV cache capacity estimator

`kv-calc` is a pure-computation tool that estimates the KV cache capacity a
model will occupy for a given request length, request count, and DP/TP
deployment. It outputs the total token count, the total KV cache capacity, and
the per-instance / per-GPU shares — for deployment capacity planning and
pre-benchmark resource estimation. It is the third item of the toolkit RFC
(#1208), tracked by #1217.

It depends only on the Python standard library, runs offline in a container,
and its capacity口径 matches `docs/source/_static/calculator.js` (the two
carry identical formulas, so results match by construction).

```bash
# list built-in preset models
ucm-toolkit run kv-calc --list

# estimate for a model, request length, request count, and DP/TP
ucm-toolkit run kv-calc \
  --model Qwen2.5-14B-Instruct \
  --input-len 4096 \
  --num-requests 1000 \
  --dp 1 \
  --tp 8

# read a local model directory's config.json
ucm-toolkit run kv-calc --model-dir /path/to/model --input-len 8192 --tp 4

# DeepSeek V4 prints both the paper formula and the measured vLLM/vLLM-Ascend
# bytes/token side by side
ucm-toolkit run kv-calc --model deepseek-v4-flash --input-len 4096 --tp 4 --num-requests 1000
```

## Per-layer KV profile

Instead of a single global `layers × kv_heads × head_dim × tokens` formula
(which is wrong for hybrid attention), `kv-calc` sums a per-layer KV profile:
each layer class contributes per-token bytes or a fixed size.

| Attention class | Per-layer contribution |
| --- | --- |
| Standard (MHA/MQA/GQA) | `2 × kv_heads × head_dim × tokens` (pure sliding-window models cap tokens at `sliding_window`) |
| MLA | `(kv_lora_rank + qk_rope_head_dim) × tokens` (no ×2; K/V compressed to a latent) |
| DSA | MLA latent + Lightning Indexer (`index_head_dim × tokens`) |
| DeepSeek V4 | per-layer `compress_ratios` (`Σ floor(T/ratio) × entry`) + sliding-window reserve + indexer; also prints the measured `bytesPerToken` |
| Mixed full/sliding (Gemma 4 / MiMo / Step) | full layers × tokens + sliding layers × `min(tokens, window)`; Gemma 4 cross-layer sharing scales by `(L - num_kv_shared_layers)/L` |
| Qwen linear/full (Gated DeltaNet) | only full-attention layers hold token-linear KV; linear layers hold a fixed recurrent state (`--include-linear-state`) |
| MiniMax MSA | standard GQA on all layers + optional Lightning Indexer side cache |

Classification is by `architectures[0]` (a curated, HF-verified registry) with a
field-inference fallback that prints an `[INFERRED]` note when the
architecture is unknown. Standard MLA/GQA are the degenerate special case,
auto-detected from the model config.

## Output

The headline block reports `tokens` (= `input-len × num-requests`), `size`
(total KV cache = `N × c`), and `per-GPU (÷ TP)`. Per-GPU divides by **TP
only**: a single request does not cross DP (DP replicates the model; each DP
rank serves a disjoint set of requests), so its KV cache lives on one DP rank
and is never scattered across DP ranks — DP does not divide the per-GPU cache.
The auxiliary "Other sizes" block additionally reports per-instance (`÷ DP`),
uniform-load per-GPU (`÷ TP×DP`), and per-request-per-GPU (`÷ TP`).

Pass `--json` for machine-readable output.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--model` | — | preset name/alias, local path, `hf://ID`, `ms://ID`, or bare `org/model` |
| `--model-dir` | — | local model directory containing `config.json` (alternative to `--model`) |
| `--source` | auto | force `preset` / `local` / `hf` / `ms` |
| `--input-len` | `1024` | tokens per request |
| `--num-requests` | `1` | concurrent request count |
| `--tp` | `1` | tensor parallelism |
| `--dp` | `1` | data parallelism |
| `--gqa-copy` | off | account for vLLM head-group replication when `num_kv_heads % TP != 0` (GQA/MHA only; MLA/DSA/V4 ignore) |
| `--kv-dtype` | bf16 | KV precision (fp8 for DeepSeek V4 nope); choices: fp32/fp16/bf16/fp8/int8/fp4 |
| `--indexer-dtype` | fp4 | indexer precision for DSA / V4 / MiniMax M3 |
| `--deployment` | vllm | which DeepSeek V4 measured deployment to highlight: `vllm` / `vllm-ascend` |
| `--include-linear-state` | off | include Qwen linear/Gated DeltaNet recurrent+conv state |
| `--presets` | — | merge extra presets from a JSON file (same flat form as built-ins; overrides by id) |
| `--list` | — | list preset models and exit |
| `--json` | off | emit JSON |
| `--verbose` | off | show model config fields and extra detail |

Boundary inputs (`--dp` / `--tp` < 1, `--num-requests` < 1, missing model
features) produce a clear error and a nonzero exit code.

## Notes

- Preset architectures, orgs, and fields are verified against the real HF
  `config.json` via `hf download`. Exceptions: Llama 3.1 (gated;
  `LlamaForCausalLM` is the canonical class) and `moonshotai/Kimi-K2` (base
  repo not separately published; fields from K2.5/K2.6).
- Models whose HF config does not expose a quantity (e.g. MiniMax-M3 MSA
  indexer `index_head_dim`, Kimi-K3 KDA recurrent state, GLM-5.2 indexer layer
  sharing) carry a `note` in the output flagging the approximation.
