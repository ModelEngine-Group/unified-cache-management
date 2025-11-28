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
#include "conn_stats.h"

namespace UC::Metrics {
    
ConnStats::ConnStats() = default;

std::string ConnStats::Name() const {
    return "ConnStats";
}

void ConnStats::Reset() {
    for (auto& v : data_) v.clear();
}

void ConnStats::Update(const std::unordered_map<std::string, double>& params) {
    for (const auto& [k, v] : params) {
        Key id = KeyFromString(k);
        if (id == Key::COUNT) continue;
        EmplaceBack(id, v);
    }
}

std::unordered_map<std::string, std::vector<double>> ConnStats::Data() {
    std::unordered_map<std::string, std::vector<double>> result;
    result["save_requests_num"] = data_[static_cast<std::size_t>(Key::save_requests_num)];
    result["save_blocks_num"] = data_[static_cast<std::size_t>(Key::save_blocks_num)];
    result["save_duration"] = data_[static_cast<std::size_t>(Key::save_duration)];
    result["save_speed"] = data_[static_cast<std::size_t>(Key::save_speed)];
    result["load_requests_num"] = data_[static_cast<std::size_t>(Key::load_requests_num)];
    result["load_blocks_num"] = data_[static_cast<std::size_t>(Key::load_blocks_num)];
    result["load_duration"] = data_[static_cast<std::size_t>(Key::load_duration)];
    result["load_speed"] = data_[static_cast<std::size_t>(Key::load_speed)];
    result["interval_lookup_hit_rates"] = data_[static_cast<std::size_t>(Key::interval_lookup_hit_rates)];
    return result;
}

Key ConnStats::KeyFromString(const std::string& k) {
    if (k == "save_requests_num")        return Key::save_requests_num;
    if (k == "save_blocks_num")          return Key::save_blocks_num;
    if (k == "save_duration")            return Key::save_duration;
    if (k == "save_speed")               return Key::save_speed;
    if (k == "load_requests_num")        return Key::load_requests_num;
    if (k == "load_blocks_num")          return Key::load_blocks_num;
    if (k == "load_duration")            return Key::load_duration;
    if (k == "load_speed")               return Key::load_speed;
    if (k == "interval_lookup_hit_rates")return Key::interval_lookup_hit_rates;
    return Key::COUNT;
}

void ConnStats::EmplaceBack(Key id, double value) {
    data_[static_cast<std::size_t>(id)].push_back(value);
}

static Registrar registrar;

}  