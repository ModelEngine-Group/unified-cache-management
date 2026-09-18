/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef AIO_AIO_IMPL_H
#define AIO_AIO_IMPL_H

#include <atomic>
#include <functional>
#include <linux/aio_abi.h>
#include <thread>

namespace aio {

class AioImpl {
public:
    struct Result {
        ssize_t nBytes;
        int32_t error;
    };
    using Callback = std::function<void(Result)>;
    struct Io {
        int32_t fd;
        uint64_t offset;
        uint32_t length;
        void* buffer;
        Callback callback;
    };

    AioImpl();
    ~AioImpl();
    void ReadAsync(Io&& io);
    void WriteAsync(Io&& io);

private:
    void CompletionLoop();
    void HarvestCompletions(std::vector<io_event>& events);
    void SubmitIo(struct iocb* cb);

    size_t queueDepth_{4096};
    size_t epollTimeoutMs{10};
    size_t batchCompleteSize{512};
    aio_context_t ctx_{0};
    int32_t eventFd_{-1};
    int32_t epollFd_{-1};
    std::atomic_bool stop_{false};
    std::thread eventThread_;
};

}  // namespace aio

#endif
