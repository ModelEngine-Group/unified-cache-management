# ConnectionManager and TransProvider Design Document

## 1. Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     AsuTransportImpl                              │
│                                                                  │
│   ┌──────────────────────┐    ┌──────────────────────────────┐  │
│   │  ConnectionManager   │    │     TransProvider             │  │
│   │                      │    │     (Abstract Base Class)     │  │
│   │  - Connection Select │    │                               │  │
│   │  - Fault Detection   │◄───│  - CreateConnection           │  │
│   │  - Endpoint Sharing  │    │  - DeleteConnections          │  │
│   │                      │    │  - Send                       │  │
│   │  ┌────────────────┐  │    │  - RegisterMemory             │  │
│   │  │ConnectionGroup │  │    │  - UnregisterMemory           │  │
│   │  │  └─ Channel[]  │  │    │  - AllocThread                │  │
│   │  │channelCache_   │  │    │  - FreeThread                 │  │
│   │  │drainList_      │  │    │                               │  │
│   │  └────────────────┘  │    └──────────┬───────────────────┘  │
│   │                      │               │                       │
│   │  RecoverLoop (BG)    │               ▼                       │
│   └──────────────────────┘    ┌──────────────────────────────┐  │
│                               │   AICPUTransProvider          │  │
│                               │   (HCOMM Implementation)      │  │
│                               │                               │  │
│                               │  endpoint_: Single HCOMM ep   │  │
│                               │  localIp_: Bound IP           │  │
│                               │  endpointRefCount_: Ref count │  │
│                               │                               │  │
│                               │  HcommEndpointCreate          │  │
│                               │  HcommChannelCreate           │  │
│                               │  HcommThreadAlloc             │  │
│                               │  HcommMemReg/Unreg            │  │
│                               │  HcommMemExport/Import        │  │
│                               └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## 2. ConnectionManager

### 2.1 Core Responsibilities

- Manage multiple ConnectionGroups (one group per remote endpoint)
- Provide connection selection strategies (Round Robin / Least Loaded)
- Detect connection faults and automatically recover
- Maintain channelCache to accelerate connection selection

### 2.2 Data Structures

```cpp
class ConnectionManager {
    // Connection groups: one group per remote endpoint
    std::vector<std::unique_ptr<ConnectionGroup>> groups_;
    
    // Active channel cache (flattened for fast selection)
    std::vector<std::shared_ptr<ConnectionChannel>> channelCache_;
    std::atomic<bool> cacheDirty_{false};
    
    // Channels pending recovery
    std::vector<std::shared_ptr<ConnectionChannel>> drainList_;
    
    // Connection creation callback
    CreateConnectionFunc createFn_;
};

class ConnectionGroup {
    std::uint32_t groupId;
    AsuEndpoint endpoint;
    std::vector<std::shared_ptr<ConnectionChannel>> channels;
};

class ConnectionChannel {
    std::uint32_t channelId;
    ConnectionGroup* group;
    TransProvider::ConnectionHandle handle_;  // void*, points to LinkContext
    std::atomic<std::uint32_t> inflightCount{0};
    std::atomic<ChannelState> state{ChannelState::ACTIVE};
    std::atomic<std::uint32_t> errorCount{0};
};
```

### 2.3 Connection Selection Strategies

**Round Robin**:
```cpp
idx = rrIndex_.fetch_add(1)
start = idx % total
for i in [0, total):
    pos = (start + i) % total
    channel = channelCache_[pos]
    if channel->state == ACTIVE && inflight < 256:
        IncrementInflight()
        return channel
```

**Least Loaded**:
```cpp
min_inflight = MAX
for channel in channelCache_:
    if channel->state == ACTIVE && inflight < min_inflight:
        min_inflight = inflight
        selected = channel
        if min_inflight == 0: break
IncrementInflight(selected)
return selected
```

**Multi-IP Load Balancing**: `channelCache_` contains channels from all groups. `SelectConnection` selects from the entire cache, naturally supporting cross-link load balancing.

### 2.4 Fault Detection and Recovery

**Fault Detection**:
```
ReportFailure(channel):
    errorCount++
    if errorCount < 2: return  // Below threshold
    MarkForDrain()  // CAS: ACTIVE → DRAINING
    cacheDirty = true
    drainList.push_back(channel)
```

**Fault Recovery (RecoverLoop, executes every 100ms)**:
```
swap(drainList_, to_recover)
for each channel in to_recover:
    if inflight > 0: put back to drainList, continue waiting
    else:
        createFn_(endpoint, 1) create new connection
        ├─ Failed → put back to drainList, retry next time
        └─ Success →
            RemoveChannel(old_channel)
            AddChannel(new_channel)
            cacheDirty = true
```

**Key Design**: Only reclaim when `inflight==0`, no timeout-based forced reclamation to avoid use-after-free.

### 2.5 Endpoint Sharing

The provider maintains only one endpoint, shared by all connections. Created on first `CreateConnection` call, subsequent calls validate `localIp` consistency:

```
endpoint_: Single HCOMM endpoint
localIp_: localIp bound to endpoint
endpointRefCount_: Reference count

CreateConnection(localIp):
    GetOrCreateEndpoint(localIp)
    ├─ endpoint_ exists and localIp matches → endpointRefCount_++
    ├─ endpoint_ exists and localIp mismatch → return error
    └─ endpoint_ is null → HcommEndpointCreate, endpointRefCount_=1
    HcommChannelCreate(endpoint_, ...)
    HcommThreadAlloc(...)

DeleteConnection(handle):
    HcommThreadFree(...)
    HcommChannelDestroy(...)
    ReleaseEndpoint(localIp) → endpointRefCount_--
       └─ When endpointRefCount_ == 0: HcommEndpointDestroy, endpoint_=nullptr
```

### 2.6 Shutdown Cleanup Order

```
ConnectionManager::Shutdown():
    1. channelCache_.clear()
    2. drainList_.clear()
    3. groups_.clear()  // Destroy ConnectionGroup last
```

Clear references first, then destroy objects to ensure all shared_ptr references are released before destruction.

### 2.7 Locks and Synchronization

| Lock | Protected Objects | Used By | Type |
|------|-------------------|---------|------|
| `structureMu_` | groups_, channelCache_ | Worker/Recover | std::shared_mutex |
| `drainMu_` | drainList_ | Worker/Poller/Recover | std::shared_mutex |

## 3. TransProvider

### 3.1 Abstract Base Class

```cpp
class TransProvider {
public:
    using ConnectionHandle = void*;
    using ThreadHandle = void*;
    using MemHandle = void*;

    // Connection management
    virtual Status CreateConnection(localIp, remoteIp, port, qpNum, timeout, &handles) = 0;
    virtual std::vector<Status> DeleteConnections(handles) = 0;

    // Data transmission
    virtual std::vector<Status> Send(ioBatches, kernelCount, quietCount) = 0;

    // Memory management
    virtual Status RegisterMemory(handle, memDescs, &memHandles) = 0;
    virtual std::vector<Status> UnregisterMemory(unregDescs) = 0;
    virtual Status GetMemTokenId(memHandle, &tokenId) = 0;

    // Thread management
    virtual Status AllocThread(threadNum, notifyNumPerThread, &threads) = 0;
    virtual std::vector<Status> FreeThread(threads) = 0;
};
```

### 3.2 AICPUTransProvider (HCOMM/RoCE Implementation)

#### Internal Structures

```cpp
struct LinkContext {
    std::string localIp;
    uint64_t channel;
    uint64_t thread;
    aclrtStream stream;
    std::string remoteIp;
    uint16_t remotePort;
};

void* endpoint_{nullptr};
std::string localIp_;
std::string ipMapPath_;
uint32_t endpointRefCount_{0};

aclrtBinHandle kernelBin_{nullptr};
aclrtFuncHandle kernelFunc_{nullptr};
std::unordered_map<std::string, uint32_t> ipToDeviceMap_;
bool ipToDeviceMapLoaded_{false};
```

#### IP-to-Device Mapping

The provider resolves remote NPU physical device IDs from IP addresses using a mapping file generated by `hccn_tool`. The file path is passed via the constructor's `ipMapPath` parameter.

```cpp
// Constructor: accepts kernel JSON path and IP map path
explicit AICPUTransProvider(const std::string& kernelJsonPath = "",
                             const std::string& ipMapPath = "");

// LoadIpToDeviceMap: reads "IP device_id" pairs from the file
void LoadIpToDeviceMap() {
    // Reads ipMapPath_ file, format: "192.168.190.170 0"
    // Populates ipToDeviceMap_ for LookupDeviceByIp()
}

// LookupDeviceByIp: returns devPhyId for a given IP
uint32_t LookupDeviceByIp(const std::string& ip) {
    // Used in CreateConnection to build remote EndpointDesc
}

// CreateConnection: builds EndpointDesc from remoteIp + IP map
Status CreateConnection(localIp, remoteIp, port, qpNum, timeout, &handles) {
    EndpointDesc remoteDesc;
    remoteDesc.loc.device.devPhyId = LookupDeviceByIp(remoteIp);
    HcommChannelCreate(endpoint, COMM_ENGINE_AICPU_TS, &channelDesc, ...)
}
```

IP map file format (one line per NPU):
```
192.168.190.170 0
192.168.190.171 1
192.168.190.172 2
...
```

Generate the IP map file before running:
```bash
for i in $(seq 0 $((NPU_COUNT-1))); do
    IP=$(hccn_tool -i ${i} -ip -g 2>/dev/null | grep "ipaddr" | awk -F: '{print $2}')
    echo "${IP} ${i}" >> /tmp/npu_ip_map.txt
done
```

#### Memory Registration/Unregistration with nullptr ConnectionHandle

`RegisterMemory` and `UnregisterMemory` accept `nullptr` as `connectionHandle` when operating on endpoint-level (not connection-level) memory. This is used in the demo for mailbox memory that needs to be registered before any connection is established:

```cpp
// Register: endpoint-level memory registration
Status RegisterMemory(nullptr, memDescs, &memHandles) {
    // When connectionHandle is nullptr, skip GetLinkContext check
    HcommMemReg(endpoint_, "asu_mem", &mem, &memHandle)
}

// Unregister: endpoint-level memory unregistration
std::vector<Status> UnregisterMemory(unregDescs) {
    // When connectionHandle is nullptr, skip GetLinkContext check
    HcommMemUnreg(endpoint_, memHandle)
}
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Engine | `COMM_ENGINE_AICPU_TS` | AICPU transport engine |
| Endpoint Type | `ENDPOINT_LOC_TYPE_DEVICE` | Device endpoint |
| Socket Role | `HCOMM_SOCKET_ROLE_RESERVED` | Reserved role |
| notifyNum | 0 | AICPU_TS doesn't need notify |
| exchangeAllMems | true | Exchange all registered memory |

#### Send Implementation

Send constructs HixlSendParam and launches the AICPU kernel via ACL API:

```cpp
std::vector<Status> Send(ioBatches, kernelCount, quietCount) {
    for each batch:
        ctx = GetLinkContext(batch.connectionHandle)
        
        // Construct HixlSendParam
        args.thread = ctx->thread
        args.channel = ctx->channel
        args.local_src = batch.sendBuffer
        args.len = batch.len
        
        // Launch kernel via ACL
        aclrtKernelArgsInit(kernelFunc_, &argsHandle)
        aclrtKernelArgsAppend(argsHandle, &args, sizeof(args), &paramHandle)
        aclrtKernelArgsFinalize(argsHandle)
        aclrtLaunchKernelWithConfig(kernelFunc_, 1, ctx->stream, &cfg, argsHandle, nullptr)
        aclrtSynchronizeStream(ctx->stream)
        
        if batch.flagBuffer:
            *flagBuffer = 1  // Mark task complete
        results.push_back(OK)
    return results
}
```

#### Memory Export/Import (Demo Only)

These interfaces are only used by the demo program for cross-process memory sharing, not used by asu_transport core logic. They accept `nullptr` as `connectionHandle` for endpoint-level operations:

```cpp
// Export: Call HcommMemExport to get descriptor
Status ExportMemory(handle, memHandle, &exportDesc, &exportLen) {
    endpoint = endpoint_
    HcommMemExport(endpoint, memHandle, &exportDesc, &exportLen)
    return OK
}

// Import: Call HcommMemImport to import remote memory
Status ImportMemory(handle, importDesc, importLen, &importedHandle) {
    endpoint = endpoint_
    CommMem outMem
    HcommMemImport(endpoint, importDesc, importLen, &outMem)
    importedMemMap_[outMem.addr] = { addr, size, memDesc }
    *importedHandle = outMem.addr
    return OK
}

// Get imported memory info
Status GetImportedMemoryInfo(handle, importedHandle, &addr, &size) {
    info = importedMemMap_[importedHandle]
    *addr = info.addr
    *size = info.size
    return OK
}
```

### 3.3 Smart Pointer Lifecycle

**ConnectionChannel**:
- Holders: `ConnectionGroup::channels`, `channelCache_`, `drainList_`, `PendingRequest::channel`
- Lifecycle: Automatically destructed when all references are released

**LinkContext**:
- Lifecycle: Freed in `DeleteConnections` via `delete ctx`

**Endpoint**:
- Holder: `AICPUTransProvider::endpoint_` (single value, shared by all connections)
- Lifecycle: `HcommEndpointDestroy` when `endpointRefCount_ == 0`

## 4. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| queryQpNum | 1 | Number of QPs for Query operations |
| loadQpNum | 2 | Number of QPs for Load operations |
| storeQpNum | 1 | Number of QPs for Store operations |
| maxInflightTasks | 64 | Maximum concurrent tasks |
| queryTimeoutMs | 5000 | Query timeout |
| loadTimeoutMs | 5000 | Load timeout |
| storeTimeoutMs | 5000 | Store timeout |
| kFailureThreshold | 2 | Failure threshold to trigger drain |
| kRecoverIntervalMs | 100 | RecoverLoop check interval |
| kMaxInflightPerChannel | 256 | Maximum inflight count per channel |

## 5. Build Instructions

> **CANN Version Requirement**: 9.1.0-beta.1

### 5.1 Dependency Installation

```bash
# Install system dependencies
dnf install -y fmt-devel spdlog-devel zlib-devel
```

Header files are included in the `trans/include/` directory:
- `acl/` — CANN ACL headers
- `hcomm/` — HCOMM communication library headers (includes `securec.h` and `securectype.h`, required because `EndpointDescInit` and `HcommChannelDescInit` in `hcomm_res_defs.h` use `memset_s`)
- `hixl_kernel/` — HixlSend kernel headers

Dynamic library dependencies (requires CANN package installation):
- `libhcomm.so` — HCOMM communication library
- `libascendcl.so` — CANN ACL runtime library

### 5.2 HIXL and HCOMM Build and Installation

> **Note**: If HIXL and HCOMM source code has not been modified, no need to rebuild and reinstall. You can skip this section.

ASU depends on HIXL and HCOMM dynamic libraries, which need to be built from source and installed to the CANN directory.

#### 5.2.1 Get Source Code

```bash
# Clone HIXL
git clone -b xxx https://gitcode.com/xxx/hixl.git /path/to/hixl

# Clone HCOMM
git clone -b xxx https://gitcode.com/xxx/hcomm.git /path/to/hcomm
```

#### 5.2.2 Build and Install HIXL

```bash
pushd /path/to/hixl
rm -rf ./build ./build_out
git pull
git checkout xxx
bash build.sh --pkg
yes | bash build_out/cann-hixl_9.1.0-beta.1_linux-aarch64.run --full --install-path=/usr/local/Ascend/cann-9.1.0-beta.1
bash build.sh --examples
popd
```

#### 5.2.3 Build and Install HCOMM

> **Important**: HCOMM requires `--full` flag to enable device-side (AICPU kernel) build, which generates `libccl_kernel.so` needed for AICPU transport.

```bash
pushd /path/to/hcomm
rm -rf ./build ./build_out
git pull
git checkout xxx
bash build.sh --pkg --full
yes | bash build_out/cann-hcomm_9.1.0-beta.1_linux-aarch64.run --full --install-path=/usr/local/Ascend/cann-9.1.0-beta.1
popd
```

**Required HCOMM Source Code Modifications**:

Two modifications are required in the HCOMM source code for AICPU transport to work correctly:

1. **AICPU Kernel Build (Device-side)**: The `--full` flag enables device-side build, producing `ccl_kernel.json` and `libccl_kernel.so` containing the AICPU kernel binaries (e.g., `RunAicpuChannelInitV3`). These are packaged as `aicpu_hcomm.tar.gz` and installed to `opp/built-in/op_impl/aicpu/kernel/`. Without `--full`, the AICPU kernel binary is missing and `HcommChannelCreate` returns error 15.

2. **HcclTaskClear Skip on AICPU**: In `src/framework/communicator/impl/independent_op/data_api/launch_context.cc`, the `HandleClear()` method must skip `HcclTaskClear` when running on AICPU (`#ifdef CCL_KERNEL_AICPU`). This is because AICPU workflows do not create graph contexts, and `HcclTaskClear` → `ResetGraphCtx` → `GetGraphCtxV2` would fail trying to find a non-existent graph context, causing `HcommBatchModeEnd` to return `HCCL_E_INTERNAL(5)`.

   Patch at `launch_context.cc:73-76`:
   ```cpp
   #ifdef CCL_KERNEL_AICPU
       HCCL_INFO("[%s] Running on AICPU, HcclTaskClear skipped.", __func__);
       return HCCL_SUCCESS;
   #else
       // original HcclTaskClear logic
   #endif
   ```

#### 5.2.4 Verify Installation

```bash
# Check HCOMM dynamic library
ls -l /usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libhcomm.so

# Check HIXL dynamic library
ls -l /usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64/libcann_hixl.so

# Check if dynamic libraries can be found by linker
ldconfig -p | grep -E "libhcomm|libcann_hixl"
```

### 5.3 ASU Build Commands

> **Important**: ASU must be built from the `kv` directory (not the `asu` directory) because it depends on `kv_common` library.

**Method 1: Using CANN Environment Variables (Recommended)**

```bash
# Source CANN environment
source /usr/local/Ascend/cann/set_env.sh

# Create build directory (must be from kv directory)
mkdir -p build_kv && cd build_kv

# Configure CMake (automatically detects CANN library path from environment variables)
cmake \
  -DUCM_ROOT_DIR=/path/to/unified-cache-management \
  -DBUILD_UCM_ASU=ON \
  -DBUILD_PROVIDER_DEMO=ON \
  /path/to/unified-cache-management/ucm/transport/kv

# Build ASU library
make asu_transport -j$(nproc)

# Build demo (requires BUILD_PROVIDER_DEMO=ON)
make aicpu_send_with_provider -j$(nproc)

# Build unit tests (optional)
make asu.test -j$(nproc)
```

**Method 2: Explicitly Specify CANN Library Path**

```bash
# Create build directory (must be from kv directory)
mkdir -p build_kv && cd build_kv

# Configure CMake
cmake \
  -DUCM_ROOT_DIR=/path/to/unified-cache-management \
  -DBUILD_UCM_ASU=ON \
  -DBUILD_PROVIDER_DEMO=ON \
  -DCANN_LIB_DIR=/usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64 \
  /path/to/unified-cache-management/ucm/transport/kv

# Build ASU library
make asu_transport -j$(nproc)

# Build demo (requires BUILD_PROVIDER_DEMO=ON)
make aicpu_send_with_provider -j$(nproc)

# Build unit tests (optional)
make asu.test -j$(nproc)
```

**Run Unit Tests:**

```bash
# Run all tests
./asu/asu.test

# Run specific test suite
./asu/asu.test --gtest_filter="ConnectionManagerTest.*"
```

**Notes:**
- **Build Directory**: Must build from `kv` directory, not `asu` directory, because ASU depends on `kv_common` library
- **Demo Build**: The demo executable `aicpu_send_with_provider` is only built when `-DBUILD_PROVIDER_DEMO=ON` is set. Without this flag, only `libasu_transport.so` is built
- **fmt Compatibility**: The project includes `common/fmt_compat.h` to provide `fmt::underlying()` for fmt 8.x (system default). This header is automatically included via CMake and does not require upgrading fmt
- CMakeLists.txt automatically detects environment variables `ASCEND_HOME_PATH` or `ASCEND_HOME` (set by `set_env.sh`)
- If environment variables exist, automatically appends `/lib64` path
- If environment variables don't exist and `CANN_LIB_DIR` is not specified, an error message will be displayed
- Method 1 is recommended as it follows the standard CANN workflow and doesn't require modifying build commands when upgrading CANN versions
- `kv_common` is built as a static library with `-fPIC` to allow linking into shared libraries

### 5.4 Build Artifacts

| Artifact | Description | Build Flag |
|----------|-------------|------------|
| `libasu_transport.so` | ASU transport layer dynamic library | `BUILD_UCM_ASU=ON` |
| `aicpu_send_with_provider` | Demo executable | `BUILD_PROVIDER_DEMO=ON` |

### 5.5 Notes

- `UCM_ROOT_DIR` points to the unified-cache-management project root directory
- CANN library path can be automatically detected from environment variables (`ASCEND_HOME_PATH` or `ASCEND_HOME`) after sourcing `set_env.sh`, or explicitly specified via `CANN_LIB_DIR` parameter
- Logger module source code is automatically compiled into `libasu_transport.so`, no separate compilation needed
- `asu_client` will fail to compile due to missing `kv_common` dependency, can be ignored
- HIXL compilation is required to generate kernel configuration files and dynamic libraries
- Ensure CANN 9.1.0-beta.1 is properly installed before compiling HCOMM

## 6. Demo Program: aicpu_send_with_provider

### 6.1 Overview

`aicpu_send_with_provider` is an end-to-end test program demonstrating how to use `AICPUTransProvider` for AICPU data transfer between two processes.

Main workflow:
1. Create AICPUTransProvider and initialize
2. Create connection and register mailbox memory
3. Export memory descriptor and exchange via file
4. Import peer memory, establish complete connection
5. Rank 0 launches HixlSend kernel to send data
6. Rank 1 launches HixlRecv kernel to receive and verify data

### 6.2 Build

The demo program is located at `test/demo/aicpu_send_with_provider.cc` and requires `-DBUILD_PROVIDER_DEMO=ON` to build:

```bash
cd /path/to/build_kv

# Build demo (automatically builds dependent asu_transport)
make aicpu_send_with_provider -j$(nproc)
```

After successful build, the executable is generated at: `/path/to/build_kv/asu/aicpu_send_with_provider`

### 6.3 Run

A `run.sh` script is provided in the `test/demo/` directory that automates the full workflow: IP map generation, launching both ranks, and log collection.

```bash
bash test/demo/run.sh
```

The script:
1. Generates `/tmp/npu_ip_map.txt` by querying each NPU's RoCE IP via `hccn_tool`
2. Starts Rank 1 (receiver) in background
3. Starts Rank 0 (sender) after a 1-second delay
4. Waits for both ranks and reports PASS/FAIL
5. Collects AICPU device logs to `test/demo/logs/`

**Manual Two-Process Launch:**

**Rank 1 (Receiver):**

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0

./aicpu_send_with_provider \
    --rank=1 \
    --logic-dev=2 \
    --phy-dev=2 \
    --ip=192.168.190.172 \
    --bytes=4096 \
    --local-file=/tmp/r1.bin \
    --peer-file=/tmp/r0.bin \
    --done-file=/tmp/hixl.done \
    --kernel-json=${ASCEND_HOME_PATH}/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json \
    --ip-map=/tmp/npu_ip_map.txt \
    --message="Hello World!"
```

**Rank 0 (Sender):**

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0

./aicpu_send_with_provider \
    --rank=0 \
    --logic-dev=0 \
    --phy-dev=0 \
    --ip=192.168.190.170 \
    --bytes=4096 \
    --local-file=/tmp/r0.bin \
    --peer-file=/tmp/r1.bin \
    --done-file=/tmp/hixl.done \
    --kernel-json=${ASCEND_HOME_PATH}/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json \
    --ip-map=/tmp/npu_ip_map.txt \
    --message="Hello World!"
```

### 6.4 Parameter Description

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--rank` | Process role (0=sender, 1=receiver) | `--rank=0` |
| `--logic-dev` | ACL logical device ID (`logicDev` must equal `phyDev`, do not set `ASCEND_RT_VISIBLE_DEVICES`) | `--logic-dev=0` |
| `--phy-dev` | NPU physical device ID (shown by npu-smi) | `--phy-dev=0` |
| `--ip` | Local RoCE NIC IP address | `--ip=192.168.190.170` |
| `--bytes` | Mailbox buffer size (bytes) | `--bytes=4096` |
| `--local-file` | Local descriptor file path | `--local-file=/tmp/r0.bin` |
| `--peer-file` | Peer descriptor file path | `--peer-file=/tmp/r1.bin` |
| `--done-file` | Completion flag file path | `--done-file=/tmp/hixl.done` |
| `--kernel-json` | HixlSend kernel configuration file (available in CANN installation) | `--kernel-json=.../libcann_hixl_kernel.json` |
| `--ip-map` | NPU IP-to-device mapping file (generated by `hccn_tool`) | `--ip-map=/tmp/npu_ip_map.txt` |
| `--message` | Message content to send | `--message="Hello World!"` |

### 6.5 Notes

1. **Startup Order**: Start Rank 1 first, then Rank 0. Rank 0 will wait for Rank 1's descriptor file to be ready. The `run.sh` script handles this automatically.

2. **Device ID Mapping**:
   - `logicDev` must equal `phyDev` — do not set `ASCEND_RT_VISIBLE_DEVICES`, which would remap logical device IDs
   - `phy-dev`: Physical NPU ID, view via `npu-smi info`

3. **IP Map Generation**:
   - Generate the IP map file before running: `run.sh` does this automatically using `hccn_tool`
   - Format: `<RoCE_IP> <device_id>` per line
   - The IP map enables `CreateConnection` to resolve `remoteIp` to the correct `devPhyId` for building the remote `EndpointDesc`

4. **Network Configuration**:
   - Both ranks' `--ip` must be RoCE NIC IPs
   - Query NPU's RoCE IP via `hccn_tool -i <device_id> -ip -g`
   - Ensure both IPs are in the same subnet and network is connected

5. **File Exchange**:
   - `--local-file` and `--peer-file` need cross-configuration
   - Rank 0's `local-file` is Rank 1's `peer-file`, and vice versa

6. **Kernel JSON**:
   - Available in CANN installation: `${ASCEND_HOME_PATH}/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json`
   - Contains `HixlSend`, `HixlRecv`, `HixlBatchGet`, `HixlBatchPut` kernel definitions

7. **Runtime Environment**:
   - Need to source CANN environment variables: `source /usr/local/Ascend/cann/set_env.sh`
   - Ensure `LD_LIBRARY_PATH` includes the directory containing `libasu_transport.so`

## 7. HCOMM Source Code Modifications

### 7.1 HcclTaskClear Skip on AICPU

**File**: `hcomm/src/framework/communicator/impl/independent_op/data_api/launch_context.cc`

**Problem**: When running on AICPU, `HcommBatchModeEnd` calls `HandleClear()` which invokes `HcclTaskClear(launchTag_)`. This tries to find a graph context via `GetGraphCtxV2`, but AICPU workflows skip `CommTaskPrepare` (which creates graph contexts) in `BatchModeStart`. The missing graph context causes `HcclTaskClear` to return `HCCL_E_PTR(5)`, propagating as `HCCL_E_INTERNAL` and failing the entire AICPU kernel (HixlRecv/HixlSend).

**Root Cause**: For HixlRecv, `HcommRecvOnThread` does not call `AddThread` (no RDMA writes), so `threadVec_` is empty. `HandleEagerMode` does not execute `CommTaskLaunch`, keeping workflow mode as `OP_BASE`. `ResetGraphCtx` then attempts to find a graph context that was never created.

**Fix**: Skip `HcclTaskClear` entirely when `CCL_KERNEL_AICPU` macro is defined (consistent with `BatchModeStart` which skips `CommTaskPrepare` on AICPU):

```cpp
// launch_context.cc, HandleClear() method
#ifdef CCL_KERNEL_AICPU
    HCCL_INFO("[%s] Running on AICPU, HcclTaskClear skipped.", __func__);
    return HCCL_SUCCESS;
#else
    DevType devType = DevType::DEV_TYPE_COUNT;
    hrtGetDeviceType(devType);
    if (devType == DevType::DEV_TYPE_950) {
        HCCL_INFO("[%s] Running on A5, HcclTaskClear skipped.", __func__);
        return HCCL_SUCCESS;
    }
    return HcclTaskClear(launchTag_);
#endif
```

**Note**: The `CCL_KERNEL_AICPU` macro is defined in `ccl_kernel.cmake` for the `ccl_kernel` target. `launch_context.cc` is compiled into both `hcomm` and `ccl_kernel` targets; only the `ccl_kernel` compilation has this macro active.

### 7.2 AICPU Kernel Binary Packaging

**Problem**: Without device-side build (`--full` flag), the HCOMM package does not include the AICPU kernel binary (`libccl_kernel.so`). This causes `HcommChannelCreate` to return error 15 (kernel not found) when attempting AICPU_TS channel creation.

**Fix**: Use `bash build.sh --pkg --full` to build with device-side support. The resulting package includes `aicpu_hcomm.tar.gz` (containing `libccl_kernel.so` + `libccl_kernel_plf.so`), which is installed to `opp/built-in/op_impl/aicpu/kernel/` and unpacked by the AICPU scheduler at runtime.

**Deployment Note**: The AICPU scheduler unpacks all `.tar.gz` files from `opp/built-in/op_impl/aicpu/kernel/` into `/usr/lib64/aicpu_kernels/<dev>/aicpu_kernels_device/` at startup. `aicpu_hcomm.tar.gz` (hcomm kernels) and `cann-hixl-compat.tar.gz` (hixl kernels) are independent and do not conflict.

### 7.3 OpenSSL Device-Side Build Fix

**Problem**: Device-side build (`--full`) fails when cmake's ExternalProject_Add for openssl invokes a nested cmake process. The host-side cmake links `libldap.so.2` which depends on `libcrypto.so.3` (OPENSSL_3.0.0), causing a symbol lookup error in the nested build environment.

**Fix**: Pre-compile openssl-3.0.9 for aarch64 using the hcc cross-compilation toolchain and place it in `third_party/openssl-device/`. The `openssl.cmake` `find_path/find_library` will detect the pre-built version and skip ExternalProject_Add, avoiding the nested cmake and libldap conflict.

**Added Files**:
- `third_party/openssl-device/lib/` (libcrypto.a, libssl.a, libcrypto.so.3, libssl.so.3)
- `third_party/openssl-device/include/openssl/` (header files)
- `openssl-device-src/include/crypto/x509.h` (cmake find_path search path)
