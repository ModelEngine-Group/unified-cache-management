#pragma once

#include "acl/acl.h"
#include "core/transport.h"

namespace transport {

class WithAclRuntimeContext {
public:
    explicit WithAclRuntimeContext(aclrtContext context);
    ~WithAclRuntimeContext();

    WithAclRuntimeContext(const WithAclRuntimeContext&) = delete;
    WithAclRuntimeContext& operator=(const WithAclRuntimeContext&) = delete;

    Status status() const { return status_; }

private:
    aclrtContext context_ = nullptr;
    aclrtContext previous_ = nullptr;
    Status status_ = Status::OK();
};

}  // namespace transport
