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
#ifndef UNIFIEDCACHE_TSF_TASK_H
#define UNIFIEDCACHE_TSF_TASK_H

#include <cstdint>
#include <string>
#include "logger/logger.h"

namespace UC {

struct TsfTask {
    enum class Type { DUMP, LOAD };         
    enum class Location { HOST, DEVICE };   
    Type type;
    Location location;  
    std::string blockId;
    size_t offset;      
    uintptr_t address;  
    size_t length;      
    size_t owner;  // taskId

    TsfTask(const Type type, const Location location, const std::string& blockId, const size_t offset,
            const uintptr_t address, const size_t length)
        : type{type}, location{location}, blockId{blockId}, offset{offset}, address{address}, length{length}, owner{0}
    {
    }
    TsfTask(const Type type, const Location location, const std::string& blockId, const size_t offset,
            const uintptr_t address, const size_t length, const size_t owner)
        : type{type}, location{location}, blockId{blockId}, offset{offset}, address{address}, length{length}, owner{owner}
    {
    }
};

class TsfTaskGroup{
    using Clock = std::chrono::steady_clock;

public:
    TsfTaskGroup():_taskId{0}, _brief{}, _number{0}, _totalSize{0}, _tp{Clock::now()}{}
    double Elapsed() const {return std::chrono::duration<double>(Clock::now() - this ->_tp).count();}
    std::string Str() const
    {
        return fmt::format("{},{},{},{}", this->_taskId, this->_brief, this->_number, this->_totalSize);
    }
    void Set(const size_t taskId, const std::string& brief, const size_t number, const size_t totalSize)
    {
        this->_taskId = taskId;
        this->_brief = brief;
        this->_number = number;
        this->_totalSize = totalSize;
    }

private:
    size_t _taskId;
    std::string _brief;
    size_t _number;
    size_t _totalSize;
    std::chrono::time_point<Clock> _tp;
};

enum class TsfTaskStatus{
    RUNNING = 0,
    FAILURE = 1,
    SUCCESS = 2,
    CANCELLED = 3,
};

} // namespace UC

#endif
