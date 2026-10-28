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
#ifndef KV_CACHE_TRANS_H
#define KV_CACHE_TRANS_H
#ifdef KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC
// UT build: pull in the test stub instead of the real Ascend ACL SDK so the
// tests can run without an Ascend NPU environment.  The stub provides the ACL
// types/functions plus the injectable failure switches in stub_control.
#include "kvcache_trans_ascend_stub.h"
#else
#include <ATen/Tensor.h>
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <atomic>
#include <dlfcn.h>
#include <iostream>
#include <kvcache_log.h>
#include <string>
#include <thread>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather.h"
#include "aclnnop/aclnn_scatter.h"

#define CHECK_ACL_KCACHE_ERROR(logger, ret, op)                                                \
    do {                                                                                       \
        if ((ret) != ACL_SUCCESS) {                                                            \
            const char *errMsg = aclGetRecentErrMsg();                                         \
            (logger).log(LogLevel::ERROR, "|KVCache Prefetch| op: %s, Error: %d, Error: %s\n", \
                         (op), (ret), errMsg ? errMsg : "Unknown error");                      \
        }                                                                                      \
    } while (0)

using aclTensor = struct aclTensor;

using _aclCreateTensor = aclTensor *(*)(const int64_t *view_dims, uint64_t view_dims_num,
                                        aclDataType data_type, const int64_t *stride,
                                        int64_t offset, aclFormat format,
                                        const int64_t *storage_dims, uint64_t storage_dims_num,
                                        void *tensor_data);

using _aclDestroyTensor = int (*)(const aclTensor *tensor);

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    return handler;
}

inline const char *GetOpApiLibName(void)
{
    return "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libopapi.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "/vllm-workspace/vllm-ascend/csrc/build/libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    return funcAddr;
}

inline void *GetCustOpApiFuncAddr(const char *apiName)
{
    static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
    if (custOpApiHandler == nullptr) { return nullptr; }
    return GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
}

inline void *GetOpApiFuncAddr(const char *apiName)
{
    auto funcAddr = GetCustOpApiFuncAddr(apiName);
    if (funcAddr != nullptr) { return funcAddr; }
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    return opApiHandler == nullptr
               ? nullptr
               : GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

int32_t CopyTrans(void **host_ptrs_addr, void **device_ptrs_addr, size_t size_bytes, size_t number,
                  std::string type, void *stream_);
void ObtainStream(void **outputStream, void **outputContext, int32_t deviceID, int32_t mStreamNum);

int32_t Synchronize(void *context, void *stream, int streamNum, Logger &mLogger);

bool IsStreamIdle(void *stream);

void DestoryStream(void *stream, void *context, int32_t deviceID, int32_t mStreamNum);

aclTensor *AtTensorToAclTensorShared(at::Tensor &tensor);

aclDataType GetAclDataType(at::ScalarType dtype);

void ReleaseAclTensor(aclTensor *tensor);
#endif
#endif