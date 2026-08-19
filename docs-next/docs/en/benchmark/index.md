# Benchmark

Reproducible UCM benchmark methodology, tooling, and reference results. This
page is a placeholder — populate it per the spec below.

## What to add

**Reader goal**: how much faster is UCM, and how do I reproduce the number
myself?

**Required content**:

- **Method**: test scenario design (prefix cache hit rate, concurrency,
  input/output length), hardware, model, framework version. Reference the test
  design already in [GLM-5.1 4-node PD](../user-guide/model-tour/index.md).
- **Usage**: how to run — script/command, dataset, parameters, reproduction
  steps.
- **Data**: standard result tables (throughput token/s, TTFT, TPOT, speedup),
  including comparison vs the "no-UCM full compute" baseline.
- **Comparison**: cross-comparison tables across stores / engines / models to
  aid selection.

**Don't**:

- Don't duplicate all raw data from detail pages here; this is an aggregation
  entry that links back to them.
- Don't mark a result "verified" unless there is a reproducible script + hardware
  record (review line 275).

**Acceptance**:

- A reader can reproduce one result by following the page.
- Result tables label their source and whether they are verified.

**Owner**: _(to be assigned)_

## Reference

- [GLM-5.1 4-node PD results](../user-guide/model-tour/index.md) — a
  worked benchmark table.
- [Compatibility](../reference/api-parameters.md) — supported models and
  platforms.
