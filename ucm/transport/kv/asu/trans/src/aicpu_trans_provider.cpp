#include "aicpu_trans_provider.h"
#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include "hcomm/hcomm_primitives.h"
#include "hcomm/hcomm_res.h"
#include "hcomm/hcomm_res_defs.h"
#include "logger.h"

namespace UC::ASU {

AICPUTransProvider::AICPUTransProvider(const std::string& kernelJsonPath,
                                       const std::string& ipMapPath)
    : ipMapPath_(ipMapPath)
{
    if (kernelJsonPath.empty()) { return; }

    aclrtBinaryLoadOptions loadOptions{};
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;

    aclError aclRet = aclrtBinaryLoadFromFile(kernelJsonPath.c_str(), &loadOptions, &kernelBin_);
    if (aclRet != ACL_SUCCESS) {
        UC_ERROR("AICPUTransProvider::AICPUTransProvider: aclrtBinaryLoadFromFile failed, ret={}",
                 aclRet);
        return;
    }

    aclRet = aclrtBinaryGetFunction(kernelBin_, "HixlSend", &kernelFunc_);
    if (aclRet != ACL_SUCCESS) {
        UC_ERROR("AICPUTransProvider::AICPUTransProvider: aclrtBinaryGetFunction failed, ret={}",
                 aclRet);
        kernelFunc_ = nullptr;
    }
}

AICPUTransProvider::~AICPUTransProvider()
{
    if (kernelBin_) {
        aclrtBinaryUnLoad(kernelBin_);
        kernelBin_ = nullptr;
    }
    kernelFunc_ = nullptr;
}

void AICPUTransProvider::LoadIpToDeviceMap()
{
    if (ipToDeviceMapLoaded_) { return; }

    std::string mapFile = ipMapPath_;
    if (mapFile.empty()) {
        ipToDeviceMapLoaded_ = true;
        return;
    }

    FILE* fp = fopen(mapFile.c_str(), "r");
    if (!fp) {
        UC_WARN("AICPUTransProvider::LoadIpToDeviceMap: failed to open {}, using default mapping",
                mapFile);
        ipToDeviceMapLoaded_ = true;
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char ip[64] = {};
        uint32_t devId = 0;
        if (sscanf(line, "%63s %u", ip, &devId) == 2) { ipToDeviceMap_[ip] = devId; }
    }
    fclose(fp);

    ipToDeviceMapLoaded_ = true;
    UC_INFO("AICPUTransProvider::LoadIpToDeviceMap: loaded {} entries from {}",
            ipToDeviceMap_.size(), mapFile);
}

uint32_t AICPUTransProvider::LookupDeviceByIp(const std::string& ip)
{
    LoadIpToDeviceMap();
    auto it = ipToDeviceMap_.find(ip);
    if (it != ipToDeviceMap_.end()) { return it->second; }
    UC_WARN("AICPUTransProvider::LookupDeviceByIp: ip={} not found, returning 0", ip);
    return 0;
}

void* AICPUTransProvider::GetOrCreateEndpoint(const std::string& localIp)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (endpoint_ != nullptr) {
        if (localIp_ != localIp) {
            UC_ERROR(
                "AICPUTransProvider::GetOrCreateEndpoint: localIp={} conflicts with existing "
                "localIp={}",
                localIp, localIp_);
            return nullptr;
        }
        endpointRefCount_++;
        return endpoint_;
    }

    EndpointDesc localDesc;
    EndpointDescInit(&localDesc, 1);
    localDesc.protocol = COMM_PROTOCOL_ROCE;
    localDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, localIp.empty() ? "0.0.0.0" : localIp.c_str(), &localDesc.commAddr.addr);
    localDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    localDesc.loc.device.devPhyId = LookupDeviceByIp(localIp);
    localDesc.loc.device.serverIdx = 0;
    localDesc.loc.device.superDevId = 0;
    localDesc.loc.device.superPodIdx = 0;

    int32_t ret = HcommEndpointCreate(&localDesc, &endpoint_);
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::GetOrCreateEndpoint: HcommEndpointCreate failed, ret={}",
                 ret);
        return nullptr;
    }

    localEndpointDesc_ = localDesc;
    localIp_ = localIp;
    endpointRefCount_ = 1;
    return endpoint_;
}

void AICPUTransProvider::ReleaseEndpoint(const std::string& localIp)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (endpoint_ == nullptr || localIp_ != localIp) { return; }

    endpointRefCount_--;
    if (endpointRefCount_ == 0) {
        HcommEndpointDestroy(endpoint_);
        endpoint_ = nullptr;
        localIp_.clear();
    }
}

Status AICPUTransProvider::CreateConnection(const std::string& localIp, const std::string& remoteIp,
                                            uint32_t port, uint32_t qpNum, uint32_t timeout,
                                            std::vector<ConnectionHandle>& connectionHandles)
{
    connectionHandles.clear();

    if (qpNum == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "qpNum must be greater than 0");
    }

    connectionHandles.reserve(qpNum);

    void* endpoint = GetOrCreateEndpoint(localIp);
    if (endpoint == nullptr) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "Failed to get/create endpoint");
    }

    EndpointDesc remoteDesc;
    EndpointDescInit(&remoteDesc, 1);
    remoteDesc.protocol = COMM_PROTOCOL_ROCE;
    remoteDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, remoteIp.c_str(), &remoteDesc.commAddr.addr);
    remoteDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    remoteDesc.loc.device.devPhyId = LookupDeviceByIp(remoteIp);
    remoteDesc.loc.device.serverIdx = 0;
    remoteDesc.loc.device.superDevId = 0;
    remoteDesc.loc.device.superPodIdx = 0;

    for (uint32_t i = 0; i < qpNum; ++i) {
        auto* ctx = new LinkContext();
        ctx->localIp = localIp;
        ctx->remoteIp = remoteIp;
        ctx->remotePort = static_cast<uint16_t>(port);

        // 创建 channel（使用共享 endpoint）
        HcommChannelDesc channelDesc;
        HcommChannelDescInit(&channelDesc, 1);
        channelDesc.remoteEndpoint = remoteDesc;
        channelDesc.port = static_cast<uint16_t>(port);
        channelDesc.notifyNum = 0;  // AICPU_TS 不需要 notify
        channelDesc.role = HCOMM_SOCKET_ROLE_RESERVED;
        channelDesc.exchangeAllMems = true;  // 交换所有注册的内存

        int32_t ret =
            HcommChannelCreate(endpoint, COMM_ENGINE_AICPU_TS, &channelDesc, 1, &ctx->channel);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::CreateConnection: ChannelCreate failed, ret={}", ret);
            delete ctx;
            ReleaseEndpoint(localIp);
            return Status::Error(StatusCode::INTERNAL_ERROR, "ChannelCreate failed");
        }

        uint32_t notifyNumPerThread = 0;
        ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &notifyNumPerThread, &ctx->thread);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::CreateConnection: ThreadAlloc failed, ret={}", ret);
            HcommChannelDestroy(&ctx->channel, 1);
            delete ctx;
            ReleaseEndpoint(localIp);
            return Status::Error(StatusCode::INTERNAL_ERROR, "ThreadAlloc failed");
        }

        aclError aclRet = aclrtCreateStream(&ctx->stream);
        if (aclRet != ACL_SUCCESS) {
            UC_ERROR("AICPUTransProvider::CreateConnection: aclrtCreateStream failed, ret={}",
                     aclRet);
            HcommThreadFree(&ctx->thread, 1);
            HcommChannelDestroy(&ctx->channel, 1);
            delete ctx;
            ReleaseEndpoint(localIp);
            return Status::Error(StatusCode::INTERNAL_ERROR, "aclrtCreateStream failed");
        }

        connectionHandles.push_back(static_cast<ConnectionHandle>(ctx));
    }

    return Status::OK();
}

std::vector<Status> AICPUTransProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& connectionHandles)
{
    std::vector<Status> results;
    results.reserve(connectionHandles.size());

    for (auto handle : connectionHandles) {
        auto* ctx = GetLinkContext(handle);
        if (!ctx) {
            results.push_back(Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid handle"));
            continue;
        }

        if (ctx->thread != 0) {
            HcommThreadFree(&ctx->thread, 1);
            ctx->thread = 0;
        }

        if (ctx->stream) {
            aclrtDestroyStream(ctx->stream);
            ctx->stream = nullptr;
        }

        if (ctx->channel != 0) {
            HcommChannelDestroy(&ctx->channel, 1);
            ctx->channel = 0;
        }

        ReleaseEndpoint(ctx->localIp);
        delete ctx;

        results.push_back(Status::OK());
    }

    return results;
}

std::vector<Status> AICPUTransProvider::Send(const std::vector<SendIoBatch>& ioBatches,
                                             uint32_t kernelCount, uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;

    std::vector<Status> results;
    results.reserve(ioBatches.size());

    // It can be changed to a batch sending interface later
    for (const auto& batch : ioBatches) {
        auto* ctx = GetLinkContext(batch.connectionHandle);
        if (!ctx) {
            results.push_back(Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid handle"));
            continue;
        }

        if (!kernelFunc_ || !ctx->stream) {
            results.push_back(
                Status::Error(StatusCode::NOT_INITIALIZED, "Kernel or stream not ready"));
            continue;
        }

        HixlSendParam args{};
        args.thread = ctx->thread;
        args.channel = ctx->channel;
        args.local_src = batch.sendBuffer;
        args.len = batch.len;

        aclrtArgsHandle argsHandle = nullptr;
        aclrtParamHandle paramHandle = nullptr;
        aclError aclRet = aclrtKernelArgsInit(kernelFunc_, &argsHandle);
        if (aclRet != ACL_SUCCESS) {
            UC_ERROR("AICPUTransProvider::Send: KernelArgsInit failed, ret={}", aclRet);
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "KernelArgsInit failed"));
            continue;
        }

        aclRet = aclrtKernelArgsAppend(argsHandle, &args, sizeof(args), &paramHandle);
        if (aclRet != ACL_SUCCESS) {
            UC_ERROR("AICPUTransProvider::Send: KernelArgsAppend failed, ret={}", aclRet);
            aclrtKernelArgsFinalize(argsHandle);
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "KernelArgsAppend failed"));
            continue;
        }

        aclRet = aclrtKernelArgsFinalize(argsHandle);
        if (aclRet != ACL_SUCCESS) {
            UC_ERROR("AICPUTransProvider::Send: KernelArgsFinalize failed, ret={}", aclRet);
            results.push_back(
                Status::Error(StatusCode::INTERNAL_ERROR, "KernelArgsFinalize failed"));
            continue;
        }

        aclrtLaunchKernelCfg cfg{};
        aclrtLaunchKernelAttr attr{};
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = 120;
        cfg.numAttrs = 1;
        cfg.attrs = &attr;

        aclRet =
            aclrtLaunchKernelWithConfig(kernelFunc_, 1, ctx->stream, &cfg, argsHandle, nullptr);
        if (aclRet != ACL_SUCCESS) {
            UC_ERROR("AICPUTransProvider::Send: LaunchKernel failed, ret={}", aclRet);
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "LaunchKernel failed"));
            continue;
        }

        aclRet = aclrtSynchronizeStream(ctx->stream);
        if (aclRet != ACL_SUCCESS) {
            UC_ERROR("AICPUTransProvider::Send: SynchronizeStream failed, ret={}", aclRet);
            results.push_back(
                Status::Error(StatusCode::INTERNAL_ERROR, "SynchronizeStream failed"));
            continue;
        }

        if (batch.flagBuffer != nullptr) {
            volatile uint32_t* flagPtr = static_cast<volatile uint32_t*>(batch.flagBuffer);
            *flagPtr = 1;
        }

        results.push_back(Status::OK());
    }

    return results;
}

Status AICPUTransProvider::RegisterMemory(ConnectionHandle connectionHandle,
                                          const std::vector<RegisterMemoryDesc>& memoryDescs,
                                          std::vector<MemHandle>& memoryHandles)
{
    if (connectionHandle != nullptr) {
        auto* ctx = GetLinkContext(connectionHandle);
        if (!ctx) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid connection handle");
        }
    }

    memoryHandles.clear();
    memoryHandles.reserve(memoryDescs.size());

    for (const auto& desc : memoryDescs) {
        CommMem mem;
        mem.type =
            (desc.memoryType == MemType::MEM_DEVICE) ? COMM_MEM_TYPE_DEVICE : COMM_MEM_TYPE_HOST;
        mem.addr = reinterpret_cast<void*>(desc.addr);
        mem.size = desc.size;

        HcommMemHandle memHandle;
        HcommResult ret = HcommMemReg(endpoint_, "asu_mem", &mem, &memHandle);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::RegisterMemory: HcommMemReg failed, ret={}", ret);
            return Status::Error(StatusCode::INTERNAL_ERROR, "HcommMemReg failed");
        }

        memoryHandles.push_back(reinterpret_cast<MemHandle>(memHandle));
    }

    return Status::OK();
}

std::vector<Status> AICPUTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& memoryDescs)
{
    std::vector<Status> results;
    results.reserve(memoryDescs.size());

    for (const auto& desc : memoryDescs) {
        if (desc.connectionHandle != nullptr) {
            auto* ctx = GetLinkContext(desc.connectionHandle);
            if (!ctx) {
                results.push_back(
                    Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid connection handle"));
                continue;
            }
        }

        HcommMemHandle memHandle = reinterpret_cast<HcommMemHandle>(desc.memoryHandle);
        HcommResult ret = HcommMemUnreg(endpoint_, memHandle);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::UnregisterMemory: HcommMemUnreg failed, ret={}", ret);
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "HcommMemUnreg failed"));
            continue;
        }

        results.push_back(Status::OK());
    }

    return results;
}

Status AICPUTransProvider::AllocThread(uint32_t threadNum,
                                       const std::vector<uint32_t>& notifyNumPerThread,
                                       std::vector<ThreadHandle>& threads)
{
    if (notifyNumPerThread.size() != threadNum) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "notifyNumPerThread size mismatch");
    }

    threads.clear();
    threads.reserve(threadNum);

    std::vector<uint64_t> hcommThreads(threadNum);
    int32_t ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, threadNum, notifyNumPerThread.data(),
                                   hcommThreads.data());

    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::AllocThread: ThreadAlloc failed, ret={}", ret);
        return Status::Error(StatusCode::INTERNAL_ERROR, "ThreadAlloc failed");
    }

    for (uint32_t i = 0; i < threadNum; ++i) {
        threads.push_back(reinterpret_cast<ThreadHandle>(hcommThreads[i]));
    }

    return Status::OK();
}

std::vector<Status> AICPUTransProvider::FreeThread(const std::vector<ThreadHandle>& threads)
{
    std::vector<Status> results;
    results.reserve(threads.size());

    std::vector<uint64_t> hcommThreads;
    hcommThreads.reserve(threads.size());
    for (auto t : threads) { hcommThreads.push_back(reinterpret_cast<uint64_t>(t)); }

    int32_t ret = HcommThreadFree(hcommThreads.data(), threads.size());
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::FreeThread: ThreadFree failed, ret={}", ret);
        for (size_t i = 0; i < threads.size(); ++i) {
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "ThreadFree failed"));
        }
        return results;
    }

    for (size_t i = 0; i < threads.size(); ++i) { results.push_back(Status::OK()); }

    return results;
}

Status AICPUTransProvider::GetMemTokenId(MemHandle memHandle, uint32_t& tokenId)
{
    if (!memHandle) { return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid memory handle"); }

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    HcommResult ret = HcommMemExport(endpoint_, memHandle, &memDesc, &memDescLen);
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::GetMemTokenId: HcommMemExport failed, ret={}", ret);
        return Status::Error(StatusCode::INTERNAL_ERROR, "HcommMemExport failed");
    }

    // 解析序列化的 ExchangeUbBufferDto 提取 tokenId
    // 序列化格式: addr(8) + size(8) + memType(4) + memTag(4+len) + tokenValue(4) + tokenId(4)
    if (memDescLen < 24) {  // 最小长度: 8+8+4+4+4+4 = 28, 但 memTag 可能为空
        UC_ERROR("AICPUTransProvider::GetMemTokenId: memDesc too short, len={}", memDescLen);
        return Status::Error(StatusCode::INTERNAL_ERROR, "Memory descriptor too short");
    }

    const uint8_t* data = static_cast<const uint8_t*>(memDesc);

    // 跳过 addr (8 bytes) + size (8 bytes) + memType (4 bytes) = 16 bytes
    size_t offset = 16;

    // 读取 memTag 长度 (4 bytes)
    if (offset + 4 > memDescLen) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "Invalid memory descriptor format");
    }
    uint32_t memTagLen = 0;
    memcpy(&memTagLen, data + offset, 4);
    offset += 4 + memTagLen;  // 跳过 memTag

    // 跳过 tokenValue (4 bytes)
    if (offset + 4 > memDescLen) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "Invalid memory descriptor format");
    }
    offset += 4;

    // 读取 tokenId (4 bytes)
    if (offset + 4 > memDescLen) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "Invalid memory descriptor format");
    }
    memcpy(&tokenId, data + offset, 4);

    return Status::OK();
}

Status AICPUTransProvider::ExportMemory(ConnectionHandle connectionHandle, MemHandle memHandle,
                                        void** exportDesc, uint32_t* exportLen)
{
    if (connectionHandle != nullptr) {
        auto* ctx = GetLinkContext(connectionHandle);
        if (!ctx) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid connection handle");
        }
    }

    if (!memHandle || !exportDesc || !exportLen) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid parameters");
    }

    HcommResult ret = HcommMemExport(endpoint_, memHandle, exportDesc, exportLen);
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::ExportMemory: HcommMemExport failed, ret={}", ret);
        return Status::Error(StatusCode::INTERNAL_ERROR, "HcommMemExport failed");
    }

    return Status::OK();
}

Status AICPUTransProvider::ImportMemory(ConnectionHandle connectionHandle, const void* importDesc,
                                        uint32_t importLen, MemHandle* importedHandle)
{
    if (connectionHandle != nullptr) {
        auto* ctx = GetLinkContext(connectionHandle);
        if (!ctx) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid connection handle");
        }
    }

    if (!importDesc || importLen == 0 || !importedHandle) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid parameters");
    }

    CommMem outMem;
    HcommResult ret = HcommMemImport(endpoint_, importDesc, importLen, &outMem);
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::ImportMemory: HcommMemImport failed, ret={}", ret);
        return Status::Error(StatusCode::INTERNAL_ERROR, "HcommMemImport failed");
    }

    MemHandle handle = reinterpret_cast<MemHandle>(outMem.addr);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ImportedMemInfo info;
        info.addr = outMem.addr;
        info.size = outMem.size;
        info.memDesc.assign(static_cast<const uint8_t*>(importDesc),
                            static_cast<const uint8_t*>(importDesc) + importLen);
        importedMemMap_[handle] = std::move(info);
    }

    *importedHandle = handle;
    return Status::OK();
}

Status AICPUTransProvider::GetImportedMemoryInfo(ConnectionHandle connectionHandle,
                                                 MemHandle importedHandle, uint64_t* addr,
                                                 uint64_t* size)
{
    if (connectionHandle != nullptr) {
        auto* ctx = GetLinkContext(connectionHandle);
        if (!ctx) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid connection handle");
        }
    }

    if (!importedHandle || !addr || !size) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid parameters");
    }

    // 从 map 中查找导入的内存信息
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = importedMemMap_.find(importedHandle);
    if (it == importedMemMap_.end()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Imported memory handle not found");
    }

    *addr = reinterpret_cast<uint64_t>(it->second.addr);
    *size = it->second.size;
    return Status::OK();
}

Status AICPUTransProvider::UnimportMemory(ConnectionHandle connectionHandle,
                                          MemHandle importedHandle)
{
    if (connectionHandle != nullptr) {
        auto* ctx = GetLinkContext(connectionHandle);
        if (!ctx) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid connection handle");
        }
    }

    if (!importedHandle) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid memory handle");
    }

    // 获取 memDesc
    std::vector<uint8_t> memDesc;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto memIt = importedMemMap_.find(importedHandle);
        if (memIt == importedMemMap_.end()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "Imported memory handle not found");
        }
        memDesc = memIt->second.memDesc;
        importedMemMap_.erase(memIt);
    }

    // 调用 HcommMemUnimport 释放导入的内存
    HcommResult ret =
        HcommMemUnimport(endpoint_, memDesc.data(), static_cast<uint32_t>(memDesc.size()));
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::UnimportMemory: HcommMemUnimport failed, ret={}", ret);
        return Status::Error(StatusCode::INTERNAL_ERROR, "HcommMemUnimport failed");
    }

    return Status::OK();
}

Status AICPUTransProvider::InitEndpoint(const std::string& localIp)
{
    void* ep = GetOrCreateEndpoint(localIp);
    if (ep == nullptr) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "Failed to create endpoint");
    }
    return Status::OK();
}

}  // namespace UC::ASU
