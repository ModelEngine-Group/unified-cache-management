/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "stats_registry.h"

namespace UC::Metrics {

StatsRegistry& StatsRegistry::GetInstance() {
    static StatsRegistry inst;          
    return inst;
}

void StatsRegistry::RegisterStats(std::string name, Creator creator) {
    auto& reg = GetInstance();         
    std::lock_guard lk(reg.mutex_);
    reg.registry_[name] = creator;
}

std::unique_ptr<IStats> StatsRegistry::CreateStats(const std::string& name) {
    auto& reg = GetInstance();
    std::lock_guard lk(reg.mutex_);
    if (auto it = reg.registry_.find(name); it != reg.registry_.end())
        return it->second();
    return nullptr;
}

std::vector<std::string> StatsRegistry::GetRegisteredStatsNames() {
    auto& reg = GetInstance();
    std::lock_guard lk(reg.mutex_);
    std::vector<std::string> names;
    names.reserve(reg.registry_.size());
    for (auto& [n, _] : reg.registry_) names.push_back(n);
    return names;
}

}   // namespace UC::Metrics