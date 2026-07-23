/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/transport.h"
#include "kv_protocol.h"
#include "pool/buffer_pool.h"
#include "status/status.h"
#include "template/spsc_ring_queue.h"

namespace transport {
class TransportManager;
}

namespace UC::DramPool {
class MetadataManager;

inline constexpr auto kThreadIdleSleepDuration = std::chrono::microseconds(100);

inline std::uint64_t SteadyNowMs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

using RequestPtr = std::unique_ptr<KvRequest>;

// Receiver keeps transport identity beside the parsed KV request.
struct RequestTask {
    RequestPtr request;
    transport::ManagerID peer_one_sided_id;
};

using RequestTaskPtr = std::unique_ptr<RequestTask>;
using RequestQueue = UC::SpscRingQueue<RequestTaskPtr>;

using TransportHandle = transport::TransferHandle;

struct TransferItem {
    // State needed to settle one item after the batch reaches terminal.
    std::uint16_t index_in_request{0};
    BlockId key{};
};

enum class DumpLoadResult : std::uint8_t {
    Ok = 0,
    Failed = 1,
};

enum class LookupResult : std::uint8_t {
    NotFound = 0,
    Exists = 1,
};

static_assert(static_cast<std::uint8_t>(DumpLoadResult::Failed) <= 0x0FU,
              "Dump/Load result codes must fit in 4 bits");

enum class CompletionStage : std::uint8_t {
    PollDataTransfer = 0,
    SubmitResponse = 1,
    PollResponseTransfer = 2,
};

struct CompletionRecord {
    CompletionStage stage{CompletionStage::PollDataTransfer};

    // State used while the request's data transfer is in flight.
    TransportHandle data_handle{transport::kInvalidTransferHandle};
    std::vector<TransferItem> transfer_items;
    std::uint64_t submit_ms{0};
    bool disconnect_attempted{false};

    // State needed to construct the request's sole response.
    KvOpcode opcode{KvOpcode::None};
    std::uint64_t response_addr{0};
    transport::ManagerID peer_one_sided_id;
    std::vector<std::uint8_t> results;

    // Ownership held from response submission until its handle reaches terminal.
    TransportHandle response_handle{transport::kInvalidTransferHandle};
    UC::BufferPool::Slot resp_buffer;
};

using CompletionQueue = UC::SpscRingQueue<CompletionRecord>;

// Non-owning runtime view. DramPoolServer owns every referenced component.
struct DramPoolRuntime {
    DramPoolRuntime(UC::DramPool::MetadataManager& metadataRef, UC::BufferPool& flagBufferPoolRef,
                    transport::TransportManager& transportRef, ProtocolManager& protocolRef,
                    RequestQueue& requestQueueRef, CompletionQueue& completionQueueRef)
        : metadata(metadataRef),
          flagBufferPool(flagBufferPoolRef),
          transport(transportRef),
          protocol(protocolRef),
          requestQueue(requestQueueRef),
          completionQueue(completionQueueRef)
    {
    }

    UC::DramPool::MetadataManager& metadata;
    UC::BufferPool& flagBufferPool;
    transport::TransportManager& transport;
    ProtocolManager& protocol;
    RequestQueue& requestQueue;
    CompletionQueue& completionQueue;
};

}  // namespace UC::DramPool
