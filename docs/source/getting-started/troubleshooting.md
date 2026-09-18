# Troubleshooting

This page lists common error codes, log messages, and troubleshooting steps for UCM.

## Error Codes

UCM uses the following error codes in its underlying C++ stores. These codes appear in log messages and exception outputs.

| Error Code | Name | Meaning | Common Cause |
|------------|------|---------|--------------|
| **0** | `OK` | Success | Operation completed normally |
| **-1** | `Error` | General error | Unclassified error, check the attached message |
| **-50000** | `InvalidParam` | Invalid parameter | Parameter is illegal, empty, or has wrong format |
| **-50001** | `OutOfMemory` | Out of memory | Memory allocation failed |
| **-50002** | `OsApiError` | OS API error | System call failed (e.g. file I/O, thread operation) |
| **-50003** | `DuplicateKey` | Duplicate key | Attempting to insert a key that already exists |
| **-50004** | `Retry` | Retry required | Transient error, retry recommended |
| **-50005** | `NotFound` | Not found | Requested object or resource does not exist |
| **-50006** | `SerializeFailed` | Serialization failed | Error during data serialization |
| **-50007** | `DeserializeFailed` | Deserialization failed | Error during data deserialization |
| **-50008** | `Unsupported` | Unsupported operation | Feature or operation not supported in current context |
| **-50009** | `NoSpace` | No space | Storage space (disk/cache) is full |
| **-50010** | `Timeout` | Timeout | Operation timed out |

## Common Issues

### Installation & Build Issues

#### `PLATFORM` environment variable not set

**Log message**:
```
WARNING: PLATFORM environment variable is not set!
Please set PLATFORM to one of: cuda, ascend, ascend-a3, musa, maca
```

**Cause**: The `PLATFORM` env var is required for building UCM. Without it, the build defaults to `simu` (simulation mode) which is only for CI testing.

**Solution**:
```bash
export PLATFORM=cuda       # For CUDA
export PLATFORM=ascend     # For Ascend A2
export PLATFORM=ascend-a3  # For Ascend A3
pip install -v -e . --no-build-isolation
```

#### Build fails with CMake errors

**Cause**: Missing build dependencies (`cmake`, `gcc`, or CUDA toolkit not in path).

**Solution**:
1. Ensure `cmake >= 3.18` is installed: `cmake --version`
2. Ensure the CUDA toolkit is installed and `nvcc` is accessible: `nvcc --version`
3. For Ascend, ensure the CANN toolkit is properly installed

#### `pip install` fails with wheel build errors

**Cause**: Building C++ extensions requires a proper compiler environment. On some systems, `--no-build-isolation` may fail if the build toolchain is not set up.

**Solution**: Use the pre-built Docker image or wheel package from [PyPI](https://pypi.org/project/uc-manager/) instead.

---

### Configuration Issues

#### Error Code -50000 (InvalidParam)

**Typical log**:
```
invalid param ... InvalidParam(...)
```

**Common causes**:
- YAML configuration file has syntax errors
- Required parameter is missing
- Parameter value type is wrong (e.g. string where integer is expected)

**Solution**:
1. Check YAML syntax: validate with `python -c "import yaml; yaml.safe_load(open('your_config.yaml'))"`
2. Verify parameter names match the [Pipeline Store](../user-guide/prefix-cache/pipeline_store.md) or [NFS Store](../user-guide/prefix-cache/nfs_store.md) documentation
3. Ensure `storage_backends` path is a valid string, not empty

#### `UCM_CONFIG_FILE` not found

**Log message**:
```
UCM config file not found: <path>
```

**Cause**: The config file path in `kv_connector_extra_config` is incorrect or the file does not exist.

**Solution**: Verify the file path exists and is accessible from the container or process. In Docker, ensure the file is mounted or copied into the container.

#### Unsupported connector type

**Log message**:
```
Unsupported connector type: <name>
```

**Cause**: The `ucm_connector_name` in the config is not a registered connector.

**Solution**: Use one of the supported connector names: `UcmPipelineStore` or `UcmNfsStore`.

#### Unknown store pipeline

**Log message**:
```
unknown store pipeline: <name>
```

**Cause**: The `store_pipeline` value is not registered in the PipelineStore registry.

**Solution**: Use a registered pipeline such as `"Cache|Posix"`, `"Cache|Ds3fs"`, `"Cache|Compress|Posix"`, etc. See `ucm/store/pipeline/connector.py` for all registered pipelines.

---

### Runtime Issues

#### Error Code -50001 (OutOfMemory)

**Common causes**:
- `cache_buffer_capacity_gb` is set too large for available `/dev/shm`
- Multiple DP instances on the same node each allocate their own CacheStore buffer
- MLA models require more shared memory

**Solution**:
1. Check system memory: `free -h`
2. Check shared memory: `df -h /dev/shm`
3. Reduce `cache_buffer_capacity_gb` in the config (default is 256 GB)
4. When running multiple DP instances on one node, reduce the value proportionally

#### Error Code -50002 (OsApiError)

**Common causes**:
- The `storage_backends` directory does not exist
- Permission denied on the storage directory
- Disk I/O error or filesystem issue

**Solution**:
1. Verify the path exists: `ls <storage_backends_path>`
2. Check write permissions: `touch <storage_backends_path>/test_file && rm <storage_backends_path>/test_file`
3. For NFS mounts, verify the mount is healthy: `mount | grep <path>`
4. Check system logs: `dmesg | tail` or `journalctl -xe`

#### Error Code -50009 (NoSpace)

**Common causes**:
- Disk space is full at the `storage_backends` path
- `posix_capacity_gb` is set larger than actual disk capacity
- GC is not enabled or not aggressive enough

**Solution**:
1. Check disk space: `df -h <storage_backends_path>`
2. Set `posix_capacity_gb` to a value matching your available disk capacity (e.g. set to 80% of actual capacity)
3. GC is automatically enabled when `posix_capacity_gb > 0` — verify it is set correctly

#### Error Code -50010 (Timeout)

**Common causes**:
- Network or storage bandwidth is abnormal (e.g. NFS mount degraded)
- System load is too high (CPU, I/O saturation)
- `timeout_ms` is set too low for the workload

**Solution**:
1. Test storage bandwidth using `vdbench` or `fio` on the mount point or local disk to verify the environment is healthy
2. Check system load: `top` or `htop`
3. Check network: `ping` and `iperf` for NFS over network
4. Increase `timeout_ms` if needed (default 30000 ms)
5. For NFS mounts, check mount options — adding `noac` or adjusting `actimeo` may help

---

### vLLM Integration Issues

#### Prefix Caching not working (no cache hits)

**Log message**:
```
request_id: xxx, total_blocks_num: N, hit hbm: 0, hit external: 0
```

**Cause**: First-time requests have no cached KV blocks. Subsequent requests with the same prefix should show hits.

**Solution**:
1. Run the same request twice — the second should hit cached blocks
2. Verify `--no-enable-prefix-caching` is not accidentally enabled (it disables vLLM's native HBM prefix cache, used only for SSD benchmarking)
3. Verify the `UCM_CONFIG_FILE` path is correct in the launch command

#### `Unsupported device platform for UCMDirectConnector`

**Cause**: The current platform is neither CUDA nor Ascend NPU.

**Solution**: UCM currently supports CUDA and Ascend platforms only. Check the [Support Matrix](../user-guide/support-matrix/support_matrix.md) for details.

#### KV dump/load errors

**Typical logs**:
```
dump kv cache failed. <error>
wait for dump kv cache failed. <error>
submit dump task failed. <error>
```

**Cause**: Store backend (Posix/NFS) encountered I/O errors during KV cache transfer.

**Solution**:
1. Check storage backend health (disk space, permissions, network)
2. Set `UC_LOGGER_LEVEL=debug` for detailed transfer logs:
   ```
   [UC][D] Cache task(...) dispatching.
   [UC][D] Posix task(...) dispatching.
   ```
3. Check if the error code matches one from the [Error Codes](#error-codes) table above

---

### Log Debugging

To view detailed UCM transfer logs, set the environment variable:

```bash
export UC_LOGGER_LEVEL=debug
```

This produces per-task logs with task IDs, operation types, data sizes, and timing:

```
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
[UC][D] Cache task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) dispatching. [PID,TID]
[UC][D] Posix task({task_id},{operation},{subtask_number},{size}) finished, cost {time}ms. [PID,TID]
```

| Field | Meaning |
|-------|---------|
| `task_id` | Unique identifier for the store task |
| `operation` | Cache Store: `DUMP` (Device→Host) or `LOAD` (Host→Device); Posix Store: `Cache2Backend` (dump to storage) or `Backend2Cache` (load from storage) |
| `subtask_number` | Number of subtasks in this operation |
| `size` | Total data size transferred in bytes |
| `cost` | Execution time in ms |

To customize the log directory:

```bash
export UCM_LOG_PATH=my_log_dir
```

By default, logs are placed under the `log` directory relative to where the vLLM service was started.