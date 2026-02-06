// Copyright (c) 2020, Huawei Technologies Co., Ltd
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
// Adapted from
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/aclnn_torch_adapter/NPUBridge.h

#pragma once
#include <c10/core/StorageImpl.h>
#include "NPUStorageImpl.h"

namespace vllm_ascend {

class NPUBridge {
public:
    // at::tensor to NPUStorageImpl
    static NPUStorageImpl* GetNpuStorageImpl(const at::Tensor& tensor);

    // c10::StorageImpl to NPUStorageImpl
    static NPUStorageImpl* GetNpuStorageImpl(c10::StorageImpl* storageImpl);

    // c10::Storage to NPUStorageImpl
    static NPUStorageImpl* GetNpuStorageImpl(c10::Storage&& storage);

    // tensor to NPUStorageDesc
    static NPUStorageDesc& GetNpuStorageImplDesc(const at::Tensor& tensor);
};
}  // namespace vllm_ascend
