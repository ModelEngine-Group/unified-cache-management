# Ds3fs Store

Use `Cache|Ds3fs` when your deployment already has a working 3FS client and you
want UCM to move KV blocks through its native client I/O API. Cache Store stages
data between device memory and host buffers. Ds3fs Store opens cache files and
submits reads and writes through `hf3fs_usrbio`.

A path on an ordinary local disk or NFS mount is insufficient for this backend.
If you only need filesystem I/O through an existing mount,
use [Cache|Posix](pipeline.md).

## Prepare the client and native library

Deploy and validate the 3FS service and client before starting UCM. Perform the
client checks as the serving user inside the serving container, including a
small write and read through the client I/O interface. UCM does not provision
the cluster or start its client.

The UCM source build creates `libds3fsstore.so` only when CMake finds both:

| Dependency | Default search location |
| --- | --- |
| `hf3fs_usrbio.h` | `/usr/local/3fs/src/lib/api` |
| `libhf3fs_api_shared` | `/usr/local/3fs/src/lib/rs/hf3fs-usrbio-sys/lib` |

The CMake cache variables are `HF3FS_USRBIO_INCLUDE_DIR` and
`HF3FS_USRBIO_LIBRARY` if you configure CMake directly. When dependencies are
missing, the build prints `ds3fsstore: Skipping build`; it can still produce a
UCM package without this backend. Before using it, verify the installed
`ucm/store/ds3fs/libds3fsstore.so` and resolve its dependencies with `ldd` on the
target Linux host.

Use [Installation](../../user-guide/quick_start/index.md) for published engine artifacts and
[Build from source](../build_from_source.md) when the
required optional library is absent. The installed 3FS headers and shared
library must match the running client; this page does not assert compatibility
with an arbitrary 3FS release.

## Configuration for Prefix Caching

The registered spelling is `Cache|Ds3fs`, including capitalization. Save this
vLLM UCM YAML after replacing the path with the prepared 3FS client path:

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Ds3fs"
      storage_backends: "/mnt/3fs/ucm-cache"
```

Use a single client mount for the initial check: Ds3fs initializes its I/O
contexts from the first `storage_backends` entry. `ior_entries` and `ior_depth`
define the 3FS rings; both default to 1. `numa_id` defaults to -1.
`stream_number` controls Ds3fs transfer workers and defaults to 32 workers. Cache device-transfer concurrency is configured
separately with `cache_stream_number`.

Keep the default ring settings until the client path works. The explicit host
budget follows [Pipeline memory sizing](pipeline.md#budget-host-memory),
including its minimum shard count. The vLLM adapter supplies block and shard
geometry. The builder uses one shard as each Ds3fs transfer unit; manually
substituting another model's size values can break the file layout.

## Original serving and request commands

Save the YAML above as `/etc/ucm/ucm.yaml` in the matching engine environment, then use the original Qwen2.5-14B-Instruct serving parameters below. `--no-enable-prefix-caching` is retained for the original storage test; remove it for normal deployments that also use native prefix caching.

```bash
export ENABLE_UCM_PATCH=1
vllm serve Qwen/Qwen2.5-14B-Instruct \
--max-model-len 20000 \
--tensor-parallel-size 2 \
--gpu_memory_utilization 0.87 \
--block_size 128 \
--trust-remote-code \
--port 7800 \
--enforce-eager \
--no-enable-prefix-caching \
--kv-transfer-config \
'{
    "kv_connector": "UCMConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/ucm.yaml"}
}'
```

After readiness, run the original request workload from another terminal. These are workload inputs, not reproduced performance results:

```bash
vllm bench serve \
--backend vllm \
--model Qwen/Qwen2.5-14B-Instruct \
--host 127.0.0.1 \
--port 7800 \
--dataset-name random \
--num-prompts 12 \
--random-input-len 16000 \
--random-output-len 2 \
--request-rate inf \
--seed 123456 \
--percentile-metrics "ttft,tpot,itl,e2el" \
--metric-percentiles "90,99" \
--ignore-eos
```

## Launch and verify the 3FS path

Use the [vLLM quickstart](../../user-guide/quick_start/index.md#vllm) or its
[Ascend counterpart](../../user-guide/quick_start/index.md#vllm-ascend), pointing
`UCM_CONFIG_FILE` at the YAML. Initialization should identify both Cache Store
and Ds3fs Store and show the intended storage path.

Validate the path in order:

1. With a fresh cache directory, submit a request that produces complete KV
   blocks. Observe completed dumps and data in the intended 3FS namespace.
2. Retain the files and restart the serving process. Replay the identical
   prompt, model revision, and KV geometry.
3. Require external-hit tokens, successful Ds3fs loads, and 3FS client read
   activity. A warm Cache Store hit does not exercise the 3FS read path.
4. Measure TTFT and throughput only after that read/write check passes.

Ds3fs currently inherits the base Store's no-op health probe. Consequently a
healthy Pipeline state for this stage does not demonstrate 3FS read/write
availability. Use actual transfer completion and client-side observations.

## Diagnose initialization and transfers

| Symptom | Inspect first |
| --- | --- |
| `libds3fsstore.so` cannot be loaded | Whether the optional target was built and the 3FS shared library is resolvable. |
| `Failed to initialize worker context` | Client mount, I/O vector/ring creation, permissions, and NUMA selection. |
| `Failed to register fd` | Whether the file belongs to the expected working 3FS client. |
| Prep, submit, or wait I/O errors | The reported 3FS return code and client/service logs. |
| Replay misses despite files | Finished writes, identical prompt/model/layout, and access to the same namespace. |

Do not apply Posix GC parameters to manage 3FS capacity: Ds3fs does not parse
those settings. Storage retention and capacity must be planned with the 3FS
service. See [troubleshooting](../../reference/troubleshooting.md) for
connector-level diagnosis.

[Implementation and extension](../extending-store.md#backend-entrypoints).
