#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "core/transport.h"
#include "hixl/hixl_types.h"

namespace hixl {
class Hixl;
}

namespace transport {

// Owns one HIXL engine and serializes its operations on a dedicated worker thread.
class HixlInstance final {
public:
    HixlInstance(Endpoint local_endpoint, int32_t device_id);
    ~HixlInstance();

    HixlInstance(const HixlInstance&) = delete;
    HixlInstance& operator=(const HixlInstance&) = delete;

    Status Initialize(const std::map<std::string, std::string>& options);
    void Finalize();

    Status RegisterMemory(const MemoryRegion& memory, hixl::MemHandle& handle);
    Status UnregisterMemory(hixl::MemHandle handle);
    Status Connect(const std::string& remote_engine, int32_t timeout_ms);
    Status Disconnect(const std::string& remote_engine, int32_t timeout_ms);
    Status TransferSync(const std::string& remote_engine, Opcode opcode,
                        const std::vector<Segment>& segments, int32_t timeout_ms);
    Status TransferAsync(const std::string& remote_engine, Opcode opcode,
                         const std::vector<Segment>& segments, hixl::TransferReq& request,
                         TransportCallTiming* timing = nullptr);
    Status GetTransferStatus(hixl::TransferReq request, TransferStatus& status,
                             TransportCallTiming* timing = nullptr);

    const Endpoint& LocalEndpoint() const;
    int32_t DeviceId() const;

private:
    using Task = std::function<Status(hixl::Hixl&)>;
    using QueuedTask = std::packaged_task<Status(hixl::Hixl&)>;

    Status Run(Task task);
    void WorkerMain(std::map<std::string, std::string> options,
                    std::promise<Status> initialize_result);
    void ProcessTasks(hixl::Hixl& engine);

    Endpoint local_endpoint_;
    int32_t device_id_ = -1;
    std::thread worker_;

    // Serializes Initialize and Finalize, including worker creation and join.
    std::mutex lifecycle_mutex_;

    // Protects tasks_, stopping_, and initialized_; cv_ coordinates state changes.
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<QueuedTask> tasks_;
    bool stopping_ = false;
    bool initialized_ = false;
};

}  // namespace transport
