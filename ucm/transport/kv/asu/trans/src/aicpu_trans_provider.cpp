#include "aicpu_trans_provider.h"
#include "logger.h"
#include <arpa/inet.h>
#include <cstring>

#ifdef UCM_USE_HCOMM
#include "hcomm_primitives.h"
#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#endif

namespace UC::ASU {

AICPUTransProvider::AICPUTransProvider() = default;

AICPUTransProvider::~AICPUTransProvider() {
    // 清理所有连接
    std::vector<ConnectionHandle> handles;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& pair : linkContexts_) {
            handles.push_back(pair.first);
        }
    }
    if (!handles.empty()) {
        DeleteConnections(handles);
    }
}

AICPUTransProvider::LinkContext* AICPUTransProvider::GetLinkContext(ConnectionHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = linkContexts_.find(handle);
    if (it == linkContexts_.end()) {
        return nullptr;
    }
    return it->second.get();
}

#ifdef UCM_USE_HCOMM
void* AICPUTransProvider::GetOrCreateEndpoint(uint32_t phyDev, const std::string& localIp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = endpointMap_.find(phyDev);
    if (it != endpointMap_.end()) {
        it->second.refCount++;
        return it->second.endpoint;
    }
    
    // 创建新的 endpoint
    EndpointDesc localDesc;
    EndpointDescInit(&localDesc, 1);
    localDesc.protocol = COMM_PROTOCOL_ROCE;
    localDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, localIp.empty() ? "0.0.0.0" : localIp.c_str(), &localDesc.commAddr.addr);
    localDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    localDesc.loc.device.devPhyId = phyDev;
    
    void* endpoint = nullptr;
    int32_t ret = HcommEndpointCreate(&localDesc, &endpoint);
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::GetOrCreateEndpoint: EndpointCreate failed, ret={}", ret);
        return nullptr;
    }
    
    endpointMap_[phyDev] = {endpoint, 1};
    return endpoint;
}

void AICPUTransProvider::ReleaseEndpoint(uint32_t phyDev) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = endpointMap_.find(phyDev);
    if (it == endpointMap_.end()) {
        return;
    }
    
    it->second.refCount--;
    if (it->second.refCount == 0) {
        if (it->second.endpoint != nullptr) {
            HcommEndpointDestroy(it->second.endpoint);
        }
        endpointMap_.erase(it);
    }
}
#endif

Status AICPUTransProvider::CreateConnection(
    const std::string& localIp,
    const std::string& remoteIp,
    uint32_t port,
    uint32_t qpNum,
    uint32_t timeout,
    std::vector<ConnectionHandle>& connectionHandles)
{
#ifdef UCM_USE_HCOMM
    connectionHandles.clear();
    connectionHandles.reserve(qpNum);

    // 从 localIp 推断 phyDev（简化处理：使用 0）
    // 实际应该从配置或 localIp 映射获取
    uint32_t phyDev = 0;
    
    // 获取或创建共享 endpoint
    void* endpoint = GetOrCreateEndpoint(phyDev, localIp);
    if (endpoint == nullptr) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "Failed to get/create endpoint");
    }

    // 创建远端 endpoint 描述
    EndpointDesc remoteDesc;
    EndpointDescInit(&remoteDesc, 1);
    remoteDesc.protocol = COMM_PROTOCOL_ROCE;
    remoteDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    inet_pton(AF_INET, remoteIp.c_str(), &remoteDesc.commAddr.addr);
    remoteDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    for (uint32_t i = 0; i < qpNum; ++i) {
        auto ctx = std::make_shared<LinkContext>();
        ctx->phyDev = phyDev;
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

        int32_t ret = HcommChannelCreate(endpoint, COMM_ENGINE_AICPU_TS, &channelDesc, 1, &ctx->channel);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::CreateConnection: ChannelCreate failed, ret={}", ret);
            ReleaseEndpoint(phyDev);
            return Status::Error(StatusCode::INTERNAL_ERROR, "ChannelCreate failed");
        }

        // 分配线程
        uint32_t notifyNumPerThread = 0;
        ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 1, &notifyNumPerThread, &ctx->thread);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::CreateConnection: ThreadAlloc failed, ret={}", ret);
            HcommChannelDestroy(&ctx->channel, 1);
            ReleaseEndpoint(phyDev);
            return Status::Error(StatusCode::INTERNAL_ERROR, "ThreadAlloc failed");
        }

        ConnectionHandle handle = reinterpret_cast<ConnectionHandle>(ctx.get());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            linkContexts_[handle] = ctx;
        }
        connectionHandles.push_back(handle);
    }

    return Status::OK();
#else
    (void)localIp;
    (void)remoteIp;
    (void)port;
    (void)timeout;
    
    connectionHandles.clear();
    connectionHandles.reserve(qpNum);
    
    for (uint32_t i = 0; i < qpNum; ++i) {
        auto ctx = std::make_shared<LinkContext>();
        ctx->phyDev = 0;
        ctx->remoteIp = remoteIp;
        ctx->remotePort = static_cast<uint16_t>(port);
        
        ConnectionHandle handle = reinterpret_cast<ConnectionHandle>(ctx.get());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            linkContexts_[handle] = ctx;
        }
        connectionHandles.push_back(handle);
    }
    
    return Status::OK();
#endif
}

std::vector<Status> AICPUTransProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& connectionHandles)
{
    std::vector<Status> results;
    results.reserve(connectionHandles.size());

    for (auto handle : connectionHandles) {
        std::shared_ptr<LinkContext> ctx;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = linkContexts_.find(handle);
            if (it == linkContexts_.end()) {
                results.push_back(Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid handle"));
                continue;
            }
            ctx = it->second;
            linkContexts_.erase(it);
        }

#ifdef UCM_USE_HCOMM
        // 释放线程
        if (ctx->thread != 0) {
            HcommThreadFree(&ctx->thread, 1);
            ctx->thread = 0;
        }

        // 销毁 channel
        if (ctx->channel != 0) {
            HcommChannelDestroy(&ctx->channel, 1);
            ctx->channel = 0;
        }

        // 释放 endpoint 引用（共享管理）
        ReleaseEndpoint(ctx->phyDev);
#endif

        results.push_back(Status::OK());
    }

    return results;
}

std::vector<Status> AICPUTransProvider::Send(
    const std::vector<SendIoBatch>& ioBatches,
    uint32_t kernelCount,
    uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;

    std::vector<Status> results;
    results.reserve(ioBatches.size());

    for (const auto& batch : ioBatches) {
        auto* ctx = GetLinkContext(batch.connectionHandle);
        if (!ctx) {
            results.push_back(Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid handle"));
            continue;
        }

#ifdef UCM_USE_HCOMM
        // AICPU_TS 模式下，Send 只是设置 flagBuffer
        // 实际的数据传输由 AICPU kernel 完成
        // 不使用 notify 机制
#endif

        // 设置 flagBuffer（供 poller 查询）
        if (batch.flagBuffer != nullptr) {
            volatile uint32_t* flagPtr = static_cast<volatile uint32_t*>(batch.flagBuffer);
            *flagPtr = 1;
        }

        results.push_back(Status::OK());
    }

    return results;
}

Status AICPUTransProvider::RegisterMemory(
    ConnectionHandle connectionHandle,
    const std::vector<RegisterMemoryDesc>& memoryDescs,
    std::vector<MemHandle>& memoryHandles)
{
    auto* ctx = GetLinkContext(connectionHandle);
    if (!ctx) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid handle");
    }

    memoryHandles.clear();
    memoryHandles.reserve(memoryDescs.size());

#ifdef UCM_USE_HCOMM
    // 获取共享 endpoint
    void* endpoint = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = endpointMap_.find(ctx->phyDev);
        if (it == endpointMap_.end()) {
            return Status::Error(StatusCode::INTERNAL_ERROR, "Endpoint not found for phyDev");
        }
        endpoint = it->second.endpoint;
    }

    for (const auto& desc : memoryDescs) {
        CommMem mem;
        mem.type = (desc.memoryType == MemType::MEM_DEVICE) ? COMM_MEM_TYPE_DEVICE : COMM_MEM_TYPE_HOST;
        mem.addr = reinterpret_cast<void*>(desc.addr);
        mem.size = desc.size;

        HcommMemHandle memHandle;
        int32_t ret = HcommMemReg(endpoint, "asu_mem", &mem, &memHandle);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::RegisterMemory: MemReg failed, ret={}", ret);
            return Status::Error(StatusCode::INTERNAL_ERROR, "MemReg failed");
        }

        memoryHandles.push_back(reinterpret_cast<MemHandle>(memHandle));
    }
#else
    for (size_t i = 0; i < memoryDescs.size(); ++i) {
        memoryHandles.push_back(reinterpret_cast<MemHandle>(i + 1));
    }
#endif

    return Status::OK();
}

std::vector<Status> AICPUTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& memoryDescs)
{
    std::vector<Status> results;
    results.reserve(memoryDescs.size());

    for (const auto& desc : memoryDescs) {
        auto* ctx = GetLinkContext(desc.connectionHandle);
        if (!ctx) {
            results.push_back(Status::Error(StatusCode::INVALID_ARGUMENT, "Invalid handle"));
            continue;
        }

#ifdef UCM_USE_HCOMM
        // 获取共享 endpoint
        void* endpoint = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = endpointMap_.find(ctx->phyDev);
            if (it == endpointMap_.end()) {
                results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "Endpoint not found for phyDev"));
                continue;
            }
            endpoint = it->second.endpoint;
        }

        MemHandle memHandle = reinterpret_cast<MemHandle>(desc.memoryHandle);
        int32_t ret = HcommMemUnreg(endpoint, memHandle);
        if (ret != 0) {
            UC_ERROR("AICPUTransProvider::UnregisterMemory: MemUnreg failed, ret={}", ret);
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "MemUnreg failed"));
            continue;
        }
#endif

        results.push_back(Status::OK());
    }

    return results;
}

Status AICPUTransProvider::AllocThread(
    uint32_t threadNum,
    const std::vector<uint32_t>& notifyNumPerThread,
    std::vector<ThreadHandle>& threads)
{
    if (notifyNumPerThread.size() != threadNum) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "notifyNumPerThread size mismatch");
    }

    threads.clear();
    threads.reserve(threadNum);

#ifdef UCM_USE_HCOMM
    std::vector<uint64_t> hcommThreads(threadNum);
    int32_t ret = HcommThreadAlloc(
        COMM_ENGINE_AICPU_TS,
        threadNum,
        notifyNumPerThread.data(),
        hcommThreads.data());

    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::AllocThread: ThreadAlloc failed, ret={}", ret);
        return Status::Error(StatusCode::INTERNAL_ERROR, "ThreadAlloc failed");
    }

    for (uint32_t i = 0; i < threadNum; ++i) {
        threads.push_back(reinterpret_cast<ThreadHandle>(hcommThreads[i]));
    }
#else
    for (uint32_t i = 0; i < threadNum; ++i) {
        threads.push_back(reinterpret_cast<ThreadHandle>(i + 1));
    }
#endif

    return Status::OK();
}

std::vector<Status> AICPUTransProvider::FreeThread(
    const std::vector<ThreadHandle>& threads)
{
    std::vector<Status> results;
    results.reserve(threads.size());

#ifdef UCM_USE_HCOMM
    std::vector<uint64_t> hcommThreads;
    hcommThreads.reserve(threads.size());
    for (auto t : threads) {
        hcommThreads.push_back(reinterpret_cast<uint64_t>(t));
    }

    int32_t ret = HcommThreadFree(hcommThreads.data(), threads.size());
    if (ret != 0) {
        UC_ERROR("AICPUTransProvider::FreeThread: ThreadFree failed, ret={}", ret);
        for (size_t i = 0; i < threads.size(); ++i) {
            results.push_back(Status::Error(StatusCode::INTERNAL_ERROR, "ThreadFree failed"));
        }
        return results;
    }
#endif

    for (size_t i = 0; i < threads.size(); ++i) {
        results.push_back(Status::OK());
    }

    return results;
}

}  // namespace UC::ASU
