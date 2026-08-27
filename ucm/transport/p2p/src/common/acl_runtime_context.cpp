/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
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
