#include "core/transport.h"

namespace transport {

InitAttrs::~InitAttrs() = default;
Transport::~Transport() = default;

Status Transport::init(const InitAttrs& options) {
    (void)options;
    return Status::Ok;
}

Status Transport::shutdown() {
    return Status::Ok;
}

Status Transport::registerMemory(const MemoryRegion& memory) {
    (void)memory;
    return Status::Ok;
}

Status Transport::unregisterMemory(const MemoryRegion& memory) {
    (void)memory;
    return Status::Ok;
}

Status Transport::exportMetadata(const ManagerID& manager_id, Metadata& out) {
    (void)manager_id;
    (void)out;
    return Status::Ok;
}

Status Transport::importMetadata(const ManagerID& manager_id, const Metadata& metadata) {
    (void)manager_id;
    (void)metadata;
    return Status::Ok;
}

Status Transport::connectPeer(const ManagerID& manager_id) {
    (void)manager_id;
    return Status::Ok;
}

Status Transport::submitTransfer(const Transfer& request) {
    (void)request;
    return Status::Ok;
}

}  // namespace transport
