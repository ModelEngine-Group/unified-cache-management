#ifndef UNIFIEDCACHE_TRANS_GDR_COPY_H
#define UNIFIEDCACHE_TRANS_GDR_COPY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "status/status.h"

enum GdrCopyKind : int {
    GdrMemcpyHostToDevice = 1,
    GdrMemcpyDeviceToHost = 2,
};

enum class GdrCompletionPollResult {
    Completed,
    Empty,
    Error,
    UnknownRequest,
};

class GdrCopyChannel {
public:
    virtual ~GdrCopyChannel() = default;

    virtual int GdrMemcpyAsync(void* dst, const void* src, size_t bytes, GdrCopyKind kind,
                               uint64_t* reqId) = 0;
    virtual GdrCompletionPollResult PollCompletion(uint64_t* reqId) = 0;
    virtual int WaitForCompletionEvent() = 0;
    virtual void InterruptCompletionWait() = 0;
};

class GdrCopyLib {
public:
    static std::shared_ptr<GdrCopyChannel> Open(int gpuId, const std::string& nicName);
    static void RegisterHostBuffer(void* host, size_t size);
    static void UnregisterHostBuffer(void* host);
    static UC::Status RegisterDeviceBuffer(void* device, size_t size);
    static void UnregisterDeviceBuffer(void* device);

private:
    GdrCopyLib() = delete;
};

#endif
