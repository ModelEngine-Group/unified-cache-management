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
#ifndef UNIFIEDCACHE_STATS_H
#define UNIFIEDCACHE_STATS_H

#include <string>
#include <unordered_map>
#include <vector>

namespace UC::Metrics {

class Stats {
public:
    explicit Stats(const std::string& name) : name_(name) {}
    std::string Name() { return name_; }
    void Update(const std::unordered_map<std::string, double>& params)
    {
        for (const auto& [key, val] : params) { data_[key].push_back(val); }
    }
    void Reset() { data_.clear(); }
    std::unordered_map<std::string, std::vector<double>> Data() { return data_; }

private:
    std::string name_;
    std::unordered_map<std::string, std::vector<double>> data_;
};

} // namespace UC::Metrics

#endif