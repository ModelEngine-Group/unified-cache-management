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
#include <map>
#include <string>
#include <vector>
#include "core/transport.h"

namespace transport {

enum class HixlRole : uint8_t {
    Server = 0,
    Client = 1,
    Bidirectional = 2,
};

struct HixlInitAttrs : public InitAttrs {
    struct Instance {
        int32_t port = 0;
        int32_t device_id = -1;
        std::map<std::string, std::string> options;
    };

    std::string ip = "127.0.0.1";
    std::vector<Instance> instances;
    HixlRole role = HixlRole::Bidirectional;
    int32_t connect_timeout_ms = 1000;
    int32_t transfer_timeout_ms = 1000;
};

}  // namespace transport
