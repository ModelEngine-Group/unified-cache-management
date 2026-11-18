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
#include "roce_process_state.h"

namespace kv {
namespace roce {

std::size_t DeviceKeyHash::operator()(const DeviceKey& key) const noexcept
{
    return std::hash<std::string>{}(key.name) ^ (static_cast<std::size_t>(key.port) << 1U) ^
           (static_cast<std::size_t>(key.gidIndex) << 9U);
}

ProcessState& ProcessState::Instance()
{
    // A function local static is initialised on first use and destroyed at
    // process exit, which is the lifetime the device and the registrations
    // need: a provider may come and go while the HCA stay open.
    static ProcessState state;
    return state;
}

}  // namespace roce
}  // namespace kv
