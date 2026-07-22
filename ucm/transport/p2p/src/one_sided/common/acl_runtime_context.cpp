#include "common/acl_runtime_context.h"
#include "logger/logger.h"

namespace transport {

WithAclRuntimeContext::WithAclRuntimeContext(aclrtContext context) : context_(context)
{
    if (context == nullptr) {
        status_ = Status::Error();
        return;
    }
    const auto get_status = aclrtGetCurrentContext(&previous_);
    if (get_status != ACL_ERROR_NONE) { previous_ = nullptr; }
    if (previous_ != context) {
        const auto set_status = aclrtSetCurrentContext(context);
        if (set_status != ACL_ERROR_NONE) {
            UC_ERROR("transport set runtime context failed: aclrtSetCurrentContext returned {}",
                     static_cast<int>(set_status));
            status_ = Status::Error();
        }
    }
}

WithAclRuntimeContext::~WithAclRuntimeContext()
{
    if (status_ == Status::OK() && previous_ != nullptr && previous_ != context_) {
        const auto status = aclrtSetCurrentContext(previous_);
        if (status != ACL_ERROR_NONE) {
            UC_ERROR("transport restore runtime context failed: aclrtSetCurrentContext returned {}",
                     static_cast<int>(status));
        }
    }
}

}  // namespace transport
