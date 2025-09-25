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
#include "cache_data.h"

namespace UC {

Status CacheData::Setup(const size_t ioSize, const size_t ioNumber)
{
    auto segmentSize = CacheSegment::TotalSize();
    this->ioNumberPerSegment_ = segmentSize / ioSize;
    auto segmentNumber = (ioNumber + this->ioNumberPerSegment_ - 1) / this->ioNumberPerSegment_;
    this->segments_.reserve(segmentNumber);
    for (size_t i = 0; i < segmentNumber; i++) {
        auto& segment = this->segments_.emplace_back();
        auto status = segment.Setup(i, ioSize);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

void* CacheData::At(const size_t index)
{
    return this->segments_[index / this->ioNumberPerSegment_].At(index % this->ioNumberPerSegment_);
}

} // namespace UC
