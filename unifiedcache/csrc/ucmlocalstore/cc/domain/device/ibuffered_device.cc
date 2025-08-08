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
#include "ibuffered_device.h"
#include <memory/memory.h>
#include "logger/logger.h"

namespace UCM {

IBufferedDevice::IBufferedDevice(const size_t bufferSize, const size_t bufferNumber)
{
    this->_bufferSize = Memory::Align(bufferSize);
    this->_bufferNumber = bufferNumber;
    this->_tmpBuffer = nullptr;
}

Status IBufferedDevice::Setup()
{
    auto buffer = Memory::AllocAlign(this->_bufferSize * this->_bufferNumber);
    if (!buffer) {
        UCM_ERROR("Failed to make buffer({},{}).", this->_bufferSize, this->_bufferNumber);
        return Status::OutOfMemory();
    }
    this->_buffer.Setup(buffer, this->_bufferSize, this->_bufferNumber);
    return Status::OK();
}

std::shared_ptr<void> IBufferedDevice::GetHostBuffer(size_t size)
{
    if (this->_buffer.Full()) {
        auto status = this->WaitFinish();
        if (status.Failure()) { return nullptr; }
        this->ResetHostBufferIndex();
    }
    if (this->_buffer.Available(size)) { return this->_buffer.GetBuffer(); }
    this->_tmpBuffer = Memory::AllocAlign(Memory::Align(size));
    return this->_tmpBuffer;
}

void IBufferedDevice::ResetHostBufferIndex() { this->_buffer.Reset(); }

}   // namespace UCM
