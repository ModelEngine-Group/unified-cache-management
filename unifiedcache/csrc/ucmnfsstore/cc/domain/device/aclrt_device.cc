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
#include "aclrt_device.h"
#include "acl/acl_rt.h"
#include "logger/logger.h"

namespace UC {

AclrtDevice::~AclrtDevice()
{
    if (!this->_stream) { return; }
    auto ret = aclrtDestroyStream(this->_stream);
    if (ret != ACL_SUCCESS) { UC_WARN("Failed({}) to run aclrtDestroyStream", ret); }
    this->_stream = nullptr;
    ret = aclrtResetDevice(this->_deviceId);
    if (ret != ACL_SUCCESS) { UC_WARN("Failed({}) to run aclrtResetDevice", ret); }
}

Status AclrtDevice::Setup()
{
    auto status = IBufferedDevice::Setup();
    if (status.Failure()){ return status; }
    auto ret = aclrtSetDevice(this->_deviceId);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed({}) to run aclrtSetDevice with device({}).", ret, this->_deviceId);
        return Status::OsApiError();
    }
    ret = aclrtCreateStream(&this->_stream);
    if (ret != ACL_SUCCESS) {
        (void)aclrtResetDevice(this->_deviceId);
        UC_ERROR("Failed({}) to run aclrtCreateStream.", ret);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status AclrtDevice::H2DAsync(void* dst, size_t dstMax, const void* src, const size_t count)
{
    auto ret = aclrtMemcpyAsync(dst, dstMax, src, count, ACL_MEMCPY_HOST_TO_DEVICE, this->_stream);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed({}) to copy data from H({}) to D({}).", ret, count, dstMax);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status AclrtDevice::D2HAsync(void* dst, size_t dstMax, const void* src, const size_t count)
{
    auto ret = aclrtMemcpyAsync(dst, dstMax, src, count, ACL_MEMCPY_DEVICE_TO_HOST, this->_stream);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed({}) to copy data from D({}) to H({}).", ret, count, dstMax);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status AclrtDevice::WaitFinish()
{
    auto ret = aclrtSynchronizeStream(this->_stream);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed({}) to synchronize device stream.", ret);
        return Status::OsApiError();
    }
    this->ResetHostBufferIndex();
    return Status::OK();
}

} // namespace UC
