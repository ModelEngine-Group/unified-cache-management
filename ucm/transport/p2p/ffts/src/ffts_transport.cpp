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
#include "ffts_transport.h"

#include <utility>

#include "detail/ffts_engine.h"

namespace UC::Transport::Ffts {

class FftsTransport::Impl {
public:
    Status Setup(int32_t deviceId) { return engine_.Setup(deviceId); }
    Status WaitEvent(void* event) { return engine_.WaitEvent(event); }
    Status Submit(const CopyDesc* copies, size_t count) { return engine_.Submit(copies, count); }
    Status Synchronize() { return engine_.Synchronize(); }

private:
    FftsEngine engine_;
};

FftsTransport::FftsTransport() : impl_(std::make_unique<Impl>()) {}

FftsTransport::~FftsTransport() = default;

FftsTransport::FftsTransport(FftsTransport&&) noexcept = default;

FftsTransport& FftsTransport::operator=(FftsTransport&&) noexcept = default;

Status FftsTransport::Setup(int32_t deviceId) { return impl_->Setup(deviceId); }

Status FftsTransport::WaitEvent(void* event) { return impl_->WaitEvent(event); }

Status FftsTransport::CopyAsync(void* dst, const void* src, size_t size)
{
    CopyDesc copy{dst, src, size};
    return Submit(&copy, 1);
}

Status FftsTransport::Submit(const CopyDesc* copies, size_t count) { return impl_->Submit(copies, count); }

Status FftsTransport::Submit(const std::vector<CopyDesc>& copies) { return Submit(copies.data(), copies.size()); }

Status FftsTransport::Synchronize() { return impl_->Synchronize(); }

}  // namespace UC::Transport::Ffts
