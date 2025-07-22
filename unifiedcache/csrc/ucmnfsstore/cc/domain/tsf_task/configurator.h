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
#ifndef UNIFIEDCACHE_CONFIGURATOR_H
#define UNIFIEDCACHE_CONFIGURATOR_H

#include <cstdint>
#include "logger/logger.h"

namespace UC {

class Configurator {
public:
    void DeviceId(const int32_t deviceId) { 
        this->_deviceId = deviceId; 
        UC_INFO("Set UC::DeviceId: {}", this->_deviceId);
    }
    int32_t DeviceId() const { 
        return this->_deviceId; 
    }
    void StreamNumber(const size_t streamNumber) { 
        this->_streamNumber = streamNumber; 
        UC_INFO("Set UC::StreamNumber: {}", this->_streamNumber);
    }
    size_t StreamNumber() const { 
        return this->_streamNumber; 
    }
    void Timeout(const size_t timeout) { 
        this->_timeout = timeout; 
        UC_INFO("Set UC::Timeout: {}", this->_timeout);
    }
    size_t Timeout() const { 
        return this->_timeout; 
    }
    void QueueDepth(const size_t queueDepth) { 
        this->_queueDepth = queueDepth; 
        UC_INFO("Set UC::QueueDepth: {}", this->_queueDepth);
    }
    size_t QueueDepth() const { 
        return this->_queueDepth; 
    }
    void BufferSize(const size_t bufferSize) { 
        this->_bufferSize = bufferSize; 
        UC_INFO("Set UC::BufferSize: {}", this->_bufferSize);
    }
    size_t BufferSize() const { 
        return this->_bufferSize; 
    }

private:
    int32_t _deviceId{-1};
    size_t _streamNumber{0};
    size_t _timeout{0};
    size_t _queueDepth{0};
    size_t _bufferSize{0}; // todo: buffer size

};
} // namespace UC {

#endif // UNIFIEDCACHE_CONFIGURATOR_H
