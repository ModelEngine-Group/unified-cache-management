#ifndef UC_INFRA_SHARDED_HANDLE_RECORDER_H
#define UC_INFRA_SHARDED_HANDLE_RECORDER_H

#include <functional>
#include <string>
#include "status/status.h"
#include <cufile.h>
#include "infra/template/hashmap.h"

namespace UC {

class CuFileHandleRecorder {
private:
    struct RecordValue {
        CUfileHandle_t handle;
        int fd;
        uint64_t refCount;
    };
    using HandleMap = HashMap<std::string, RecordValue, std::hash<std::string>, 10>;
    HandleMap handles_;
    CuFileHandleRecorder() = default;
    CuFileHandleRecorder(const CuFileHandleRecorder&) = delete;
    CuFileHandleRecorder& operator=(const CuFileHandleRecorder&) = delete;

public:
    static CuFileHandleRecorder& Instance()
    {
        static CuFileHandleRecorder recorder;
        return recorder;
    }

    Status Get(const std::string& path, CUfileHandle_t& handle,
               std::function<Status(CUfileHandle_t&, int&)> instantiate)
    {
        auto result = handles_.GetOrCreate(path, [&instantiate](RecordValue& value) -> bool {
            int fd = -1;
            CUfileHandle_t h = nullptr;

            auto status = instantiate(h, fd);
            if (status.Failure()) {
                return false;
            }

            value.handle = h;
            value.fd = fd;
            value.refCount = 1;
            return true;
        });

        if (!result.has_value()) {
            return Status::Error();
        }

        auto& recordValue = result.value().get();
        recordValue.refCount++;
        handle = recordValue.handle;
        return Status::OK();
    }

    void Put(const std::string& path,
             std::function<void(CUfileHandle_t)> cleanup)
    {
        handles_.Upsert(path, [&cleanup](RecordValue& value) -> bool {
            value.refCount--;
            if (value.refCount > 0) {
                return false;
            }
            cleanup(value.handle);
            return true;
        });
    }

    void ClearAll(std::function<void(CUfileHandle_t, int)> cleanup)
    {
        handles_.ForEach([&cleanup](const std::string& path, RecordValue& value) {
            cleanup(value.handle, value.fd);
        });
        handles_.Clear();
    }
};

} // namespace UC

#endif // UC_INFRA_SHARDED_HANDLE_RECORDER_H
