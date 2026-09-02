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

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/ub/status.h"

namespace umc::comm {

struct LocalJettyHandle;

namespace detail {

struct AivQueueAliasPlan {
    std::vector<std::size_t> logicalToPhysical;
    std::vector<const LocalJettyHandle*> physicalLocals;
};

inline UbStatus BuildAivQueueAliasPlan(const std::vector<const LocalJettyHandle*>& locals,
                                       AivQueueAliasPlan* out)
{
    if (out == nullptr) {
        return UbStatus(UbErrorCode::InvalidArgument, "BuildAivQueueAliasPlan: out == nullptr");
    }
    out->logicalToPhysical.clear();
    out->physicalLocals.clear();
    out->logicalToPhysical.reserve(locals.size());
    out->physicalLocals.reserve(locals.size());

    std::unordered_map<const LocalJettyHandle*, std::size_t> physicalByLocal;
    physicalByLocal.reserve(locals.size());
    for (std::size_t logical = 0; logical < locals.size(); ++logical) {
        const LocalJettyHandle* local = locals[logical];
        if (local == nullptr) {
            out->logicalToPhysical.clear();
            out->physicalLocals.clear();
            return UbStatus(
                UbErrorCode::InvalidArgument,
                "BuildAivQueueAliasPlan: null local at logical slot " + std::to_string(logical));
        }
        const auto [it, inserted] = physicalByLocal.emplace(local, out->physicalLocals.size());
        if (inserted) { out->physicalLocals.push_back(local); }
        out->logicalToPhysical.push_back(it->second);
    }
    return UbStatus::Ok();
}

}  // namespace detail
}  // namespace umc::comm
