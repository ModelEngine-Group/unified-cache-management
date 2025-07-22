/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_IDEVICE
#define UNIFIEDCACHE_IDEVICE

#include <memory>
#include "idevice.h"
#include "status/status.h"

namespace UC {

class BufferDevice : public IDevice {   // todo

public:
    BufferDevice(int32_t deviceId, size_t bufferSize, size_t bufferNumber)
            : deviceId_(deviceId), bufferSize_(bufferSize), bufferNumber_(bufferNumber) {}
    Status Setup() override
    {
        return Status::OK();
    }
private:
    int32_t deviceId_;
    size_t bufferSize_;
    size_t bufferNumber_;
};

class Device {
public:
    static std::unique_ptr<IDevice> Make(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
    {
        try{
            return std::make_unique<BufferDevice>(deviceId, bufferSize, bufferNumber);
        } catch (const std::exception& e) {
            return nullptr;
        }
    }
};

} // namespace UC

#endif // UNIFIEDCACHE_IDEVICE