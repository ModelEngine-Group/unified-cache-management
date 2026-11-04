# Device-less verification: acceptance criteria and evidence

What a device-less run has to produce before its PASS may be used as evidence,
and how to re-run it. The tool itself is documented in [README.md](README.md);
this file is about trusting its output.

## The two claims, and why their order matters

A device-less run makes one claim:

> The dump/load result obtained with no accelerator and no checkpoint weights
> is the same result a real card would have produced.

That splits into two independent questions with different evidence:

| question | asks | established by |
|---|---|---|
| **faithfulness** | is the KV layout UCM sees here byte-identical to the one a real card produces? | the real-card cross-check ([below](#faithfulness-the-real-card-cross-check)) |
| **correctness** | does UCM's dump/load have a bug? | black-box three-state judgement, stray-write check, fault injection ([below](#correctness-what-a-pass-actually-asserts)) |

A device-less run can **substitute** for a real card on the second question
only once the first has been established for that launcher and that model
class. The order cannot be reversed: a faithful-looking comparison between two
runs that both skipped the check would agree perfectly.

## Running it

Build UCM with **`PLATFORM` unset** so transfers go through the `simu` runtime
(host malloc/memcpy, no device API):

```
unset PLATFORM
pip install -v -e . --no-build-isolation --no-deps
pip install -e ./toolkit --no-deps --no-build-isolation
pip install wrapt==1.17.2
```

`--no-build-isolation` and the explicit `wrapt` are not optional on the vendor
images: isolated builds try to download setuptools and fail, and UCM's patch
helpers import `wrapt`, which those images do not ship.

Confirm the build is the `simu` one before trusting any result from it:

```
for so in $(find ucm -name '*.so'); do
    echo "$so cudart_refs=$(nm -D "$so" 2>/dev/null | grep -c cudart)"
done
```

Every line must read `cudart_refs=0`. A non-zero count means a real-device
build tree got mixed in and the run is void.

Then, per platform:

```
# Ascend layout, no NPU: mount the CANN driver read-only, map no /dev/davinci*
./run_model_check.sh /path/to/model --platform ascend-sim --ascend-soc A2

# Official layout, no GPU
./run_model_check.sh /path/to/model --platform cuda-sim --cuda-capability 12.0
```

`--ascend-soc` and `--cuda-capability` are **declarations, not guesses**. vLLM
branches on both (FlashAttention version, fp8 support, MLA backend choice), and
neither can be read without a device, so they must be given and must match the
card being simulated. The launchers assert the declaration against the
installed wheel where they can and refuse to infer it where they cannot.

Self-evidence that there really was no device, captured in the run:

```
# ascend-sim
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

Must print `False 0`. If it prints `True`, that run used a card.

## Correctness: what a PASS actually asserts

**A zero exit status is not evidence.** Each case must show the checker's own
`PASS` line. This is not hypothetical: a missing `__main__` guard in
`ucm_toolkit.cli` once produced exit 0 with an empty log and no checks run.

| case | required line |
|---|---|
| `layout` | `PASS: layout-only (UCM parsed the native KV layout)`, with the spec and group lines above it |
| `bulk` | `PASS: Scheduler->UCM dump/load, ...` |
| `layerwise` | `PASS: Scheduler->UCM dump/load, ...` |

Lines that must **not** appear — each means checks were skipped while the run
still reported PASS, which is more dangerous than a FAIL:

| line | what it means |
|---|---|
| `PARTIALLY VERIFIED` | that group has no independent required-restore set, so its PASS is not completeness. Expected only for Mamba state groups and non-whole-block sliding windows |
| `hit check skipped` | the checker did not find the v2 worker internals, so every v2 check was bypassed |
| `stray_write_check=off` | writes outside the allowed ranges were not detected |

### Why the judgement is three-state

The black-box check fills the KV region with pattern bytes drawn from
`[0, 250]`, dumps, overwrites the whole region with **poison `0xff`**, loads,
and then classifies every byte:

- equals the pattern → **correct**
- equals `0xff` → **untouched** (never restored)
- anything else → **bad** (restored to the wrong value)

Capping the pattern at 250 is what makes the third state distinguishable: a
poison byte can never be a legal pattern byte. A two-state pass/fail would lump
"not written" together with "written wrong", which are different bugs.

The pattern is also multi-scale, period 251³ ≈ 15.8 MB. An earlier single-scale
version had a 251-byte period, under which a whole-block offset error of
exactly 251 bytes was invisible.

`transfer_ranges_checked` is a separate, independent assertion: every transfer
address the connector reports is checked against the ranges derived from vLLM's
own block tables. This must run **before** poisoning — poisoning uses those
same addresses, so poisoning first and checking after would be self-confirming.

### Fault injection

The three counters above only show that writes landed where they should. Only
injecting a fault shows the check has teeth. This found a real gap once: an
injected fault was **not** caught, which revealed the strict oracle covered 1
of 11 persistent groups.

Not yet in CI. It remains the single most valuable of the four criteria below.

## Faithfulness: the real-card cross-check

Run the same model on the device-less launcher and on a real card, and compare:

1. **`sim.json` md5 identical** — the layout snapshot (shapes, strides, groups)
2. **normalized log diff empty** for every case — normalization strips
   timestamps, pids and temporary paths
3. **counters reproduce from first principles** — derivable from layer and
   block counts alone, not read back from the run
4. **an injected fault is caught, with its location named**

Criteria 1–3 only establish that two runs agree. Criterion 4 is the one that
establishes they agree *on something real*.

Established so far: `cuda_sim` is faithful for Qwen3-0.6B at capability 12.0
(RTX 5090). All four criteria passed; the three `sim.json` files were md5
identical and all normalized log diffs were empty.

**Not established:** that `--cuda-capability` changes anything for this model.
The attempted control — run 9.0 against 12.0 and expect different layouts —
is inconclusive, because Qwen3-0.6B's layout is identical under both. Proving
that flag has an effect needs a model whose layout branches on capability.

## Baseline counters (Qwen3-0.6B, 1024 tokens, block size 128)

Structural values, not samples; a re-run should reproduce them.

| counter | `cuda_sim` (5090) | `ascend_sim` (910B) | derivation |
|---|---|---|---|
| hit check | FA dumped 8/8, loaded 7/7 | same | 1024 / 128 = 8 dumped; the load side can only hit the first 7 |
| restored | 102760448 / 117440512 (87.5%) | **identical** | 7/8 |
| `compared_loaded_tensor_blocks` | 196 | 196 | 28 layers × 7 blocks |
| `poisoned_blocks` | 224 | 448 | 28 × 8; Ascend ×2, see below |
| `strict_oracle_blocks` | 196 | 392 | Ascend ×2, see below |
| `transfer_ranges_checked` | 196 | 196 | |
| `blackbox_groups` | 1 | 1 | one full-attention group |

The Ascend doubling is granularity, not a discrepancy. `poison_all_blocks`
counts `(group, layer, part, index)` entries, where `part` is the K/V split:
vLLM-Ascend gives each layer two separate KV tensors (part = 2), official vLLM
one `[2, …]` tensor (part = 1). The decisive check is the byte total, which is
identical on both platforms — same bytes, counted at a different granularity.

## Evidence a run must produce

One directory per host. A run missing any item is incomplete:

```
<platform>-<timestamp>/
├── build-info.txt     # commit, PLATFORM unset, per-.so cudart_refs, PYTHONPATH
├── nocard-proof.txt   # the is_available / device_count output above
├── run/
│   ├── layout-only.log
│   ├── bulk.log
│   └── layerwise.log
├── sim.json
└── console.log
```

If the run needed any local patch, it must also carry the patch and proof that
it was necessary and layout-neutral:

```
├── local-patches-applied.diff
├── attempt1-nopatch/        # the failure without it
├── json-diff-vs-old.txt
└── norm-{bulk,layerwise,layout-only}-vs-old.diff
```

Quote log lines verbatim rather than summarizing them. The inconclusive
capability control above was once written up as "passed".

## Known gaps

Do not read a PASS as covering these:

- **Mamba state groups and non-whole-block sliding windows** have no
  independent required-restore set; they are reported `PARTIALLY VERIFIED`.
- **Fault injection is manual**, not in CI.
- **The worker hook sequence is hand-rolled**, not driven by a real engine
  loop. This is the baseline design's known cost.
- **Device-less verification does not replace service verification.** It
  establishes that the connector's dump/load bytes are correct; it says nothing
  about a real engine serving end to end. The two are complementary.
