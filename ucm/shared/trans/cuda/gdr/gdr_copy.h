#ifndef UNIFIEDCACHE_TRANS_GDR_COPY_H
#define UNIFIEDCACHE_TRANS_GDR_COPY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum GdrCopyKind : int {
    GdrMemcpyHostToDevice = 1,
    GdrMemcpyDeviceToHost = 2,
};

class GdrCopyChannel {
public:
    virtual ~GdrCopyChannel() = default;

    virtual int GdrMemcpyAsync(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                               uint64_t* reqId) = 0;
    virtual int PollCompletion(uint64_t* reqId) = 0;
};

class GdrCopyLib {
public:
    static std::shared_ptr<GdrCopyChannel> Open(int gpuId, const std::string& nicName);
    static void RegisterHostBuffer(void* host, size_t size);
    static void UnregisterHostBuffer(void* host);
    static void RegisterDeviceBuffer(void* device, size_t size);
    static void UnregisterDeviceBuffer(void* device);

private:
    GdrCopyLib() = delete;
};

#endif
