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
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include "kv_types.h"

namespace kv {

struct ViewHttpRequest {
    // Missing since_epoch is represented by zero, which forces a full view.
    std::uint64_t sinceEpoch{0};
};

enum class ViewHttpReplyKind {
    NOT_MODIFIED,
    FULL_VIEW,
    INTERNAL_ERROR,
};

struct ViewHttpReply {
    ViewHttpReplyKind kind{ViewHttpReplyKind::INTERNAL_ERROR};
    // Used only by FULL_VIEW; the server calculates Content-Length.
    std::string body;
};

using ViewHttpHandler = std::function<ViewHttpReply(const ViewHttpRequest&)>;

// Minimal read-only HTTP/1.1 server contract. The implementation accepts only
// GET /v1/views, parses since_epoch, maps the reply kind to 304/200/500, and
// closes each connection after writing the response.
class ViewHttpServer {
public:
    virtual ~ViewHttpServer() = default;

    virtual Status Start(const std::string& listenAddr, const ViewHttpHandler& handler) = 0;
    virtual void Stop() = 0;
};

}  // namespace kv
