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
#ifndef UNIFIEDCACHE_LATCH_H
#define UNIFIEDCACHE_LATCH_H

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace UC {

class Latch {
public:
    explicit Latch(const size_t expected = 0) : _counter{expected} {}
    void Up() { ++this->_counter; }
    size_t Done() { return --this->_counter; }
    void Notify() { this->_cv.notify_all(); }
    void Wait()
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        if (this->_counter == 0) { return; }
        this->_cv.wait(lk, [this] { return this->_counter == 0; });
    }

private:
    std::mutex _mutex;
    std::condition_variable _cv;
    std::atomic<size_t> _counter;
};

} // namespace UC

#endif // UNIFIEDCACHE_LATCH_H
