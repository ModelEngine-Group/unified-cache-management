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
#include "kvcache_trans_ascend.h"
#define ACL_MEMCPY_ASYNC_TIMEOUT_TIME 200

using Ptr = uintptr_t;

int32_t CopyTrans(void** host_ptrs_addr, void** device_ptrs_addr, size_t size_bytes, size_t number,
                  std::string type, void* stream_)
{
    aclError ret = ACL_SUCCESS;
    aclrtStream input_stream = *(reinterpret_cast<aclrtStream*>(stream_));
    for (size_t i = 0; i < number; i++) {
        if (type == "h2d") {
            ret = aclrtMemcpyAsync(device_ptrs_addr[i], size_bytes, host_ptrs_addr[i], size_bytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE, input_stream);
        } else {
            ret = aclrtMemcpyAsync(host_ptrs_addr[i], size_bytes, device_ptrs_addr[i], size_bytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, input_stream);
        }
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            std::cout << "aclrtMemcpyAsync failed, trans type: " << type.c_str()
                      << " error: " << ret << " index: " << i << "/" << number
                      << " device_ptrs_addr: " << device_ptrs_addr[i]
                      << " host_ptrs_addr: " << host_ptrs_addr[i] << " error: " << errMsg
                      << std::endl;
            return -1;
        }
    }
    return 0;
}

void ObtainStream(void** outputStream, void** outputContext, int32_t deviceID, int32_t mStreamNum)
{
    auto ret = aclrtSetDevice(deviceID);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string("aclrtSetDevice failed: ") + std::to_string(ret));
    }

    aclrtContext* context = (aclrtContext*)malloc(sizeof(aclrtContext));
    ret = aclrtCreateContext(context, deviceID);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string("aclrtCreateContext failed: ") + std::to_string(ret));
    }

    aclrtStream* stream_ = (aclrtStream*)malloc(sizeof(aclrtStream) * mStreamNum);
    for (int i = 0; i < mStreamNum; i++) {
        ret = aclrtCreateStream(stream_ + i);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(std::string("AscendStreamCreateWithFlags failed: ") +
                                     std::to_string(ret));
        }
    }

    *outputStream = static_cast<void*>(stream_);
    *outputContext = static_cast<void*>(context);

    // std::cout << "========================================" << std::endl;
    // std::cout << "create ascend streams done !!!!" << stream_ << std::endl;
    // std::cout << "========================================" << std::endl;
}

int32_t Synchronize(void* context, void* stream, int streamNum, Logger& mLogger)
{
    auto ret = aclrtSetCurrentContext(*(static_cast<aclrtContext*>(context)));
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        mLogger.log(LogLevel::ERROR,
                    "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s\n", ret,
                    errMsg ? errMsg : "Unknown error");
        return -1;
    }
    for (int i = 0; i < streamNum; i++) {
        ret = aclrtSynchronizeStream((static_cast<aclrtStream*>(stream))[i]);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            mLogger.log(LogLevel::ERROR,
                        "|KVCache Prefetch| aclrtSetCurrentContext Error: %d, Error: %s\n", ret,
                        errMsg ? errMsg : "Unknown error");
            return -1;
        }
    }
    return 0;
}

bool IsStreamIdle(void* stream)
{
    aclrtStreamStatus status = ACL_STREAM_STATUS_RESERVED;
    aclrtStreamQuery(*(static_cast<aclrtStream*>(stream)), &status);
    if (status == ACL_STREAM_STATUS_COMPLETE) {
        return true;
    } else if (status == ACL_STREAM_STATUS_NOT_READY) {
        return false;
    } else {
        std::cerr << "Error querying stream: " << status << std::endl;
        return false;
    }
}

void DestroyStream(void* stream, void* context, int32_t deviceID, int32_t mStreamNum)
{
    for (int i = 0; i < mStreamNum; i++) {
        auto ret = aclrtDestroyStream((static_cast<aclrtStream*>(stream))[i]);
        if (ret != ACL_SUCCESS) { std::cout << "aclrtDestroyStream failed: " << ret << std::endl; }
    }
    free(stream);
    aclrtDestroyContext(*(static_cast<aclrtContext*>(context)));
    free(context);
    aclrtResetDevice(deviceID);
    std::cout << "====================================================" << std::endl;
    std::cout << "Destroy ascend streams and context done! stream: " << stream
              << " context: " << context << std::endl;
    std::cout << "====================================================" << std::endl;
}

aclDataType GetAclDataType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kFloat: return ACL_FLOAT;
        case at::kBFloat16: return ACL_BF16;
        case at::kUInt64: return ACL_UINT64;
        case at::kUInt32: return ACL_UINT32;
        case at::kHalf: return ACL_FLOAT16;
        case at::kInt: return ACL_INT32;
        case at::kLong: return ACL_INT64;
        case at::kDouble: return ACL_DOUBLE;
        case at::kBool: return ACL_BOOL;
        case at::kChar: return ACL_INT8;
        default: return ACL_DT_UNDEFINED;
    }
}

aclTensor* AtTensorToAclTensorShared(at::Tensor& tensor)
{
    static const auto aclCreateTensor =
        GET_OP_API_FUNC(aclCreateTensor);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    aclDataType acl_dtype = GetAclDataType(tensor.scalar_type());
    c10::SmallVector<int64_t, 5> storageDims;
    auto itemsize = tensor.itemsize();
    storageDims.push_back(tensor.storage().nbytes() / itemsize);
    auto acl_tensor = aclCreateTensor(tensor.sizes().data(), tensor.sizes().size(), acl_dtype,
                                      tensor.strides().data(), tensor.storage_offset(),
                                      ACL_FORMAT_ND, storageDims.data(), storageDims.size(),
                                      const_cast<void*>(tensor.storage().data()));
    return acl_tensor;
}

void ReleaseAclTensor(aclTensor* tensor)
{
    static const auto aclDestroyTensor =
        GET_OP_API_FUNC(aclDestroyTensor);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    if (aclDestroyTensor == nullptr) { return; }
    aclDestroyTensor(tensor);
}