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
#ifndef KV_SEMANTICS_TESTS_FAKE_HCCP_KV_OVER_ROCE_ABI_H
#define KV_SEMANTICS_TESTS_FAKE_HCCP_KV_OVER_ROCE_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Control surface exported by every fake ABI variant. The unit tests dlopen the
 * fake library, resolve these symbols and use them to program the behavior of
 * the provider-facing ABI. The state is per loaded library, so a variant switch
 * must also reopen the library. */

void HccpFakeReset(void);

void HccpFakeSetCreateResult(int32_t result);
void HccpFakeSetCreateReportedCount(uint32_t count, int32_t useQpNum);
void HccpFakeSetCreateManagerNull(int32_t managerNull);
void HccpFakeSetDeleteResult(int32_t result);
void HccpFakeSetRegisterResult(int32_t result);
void HccpFakeSetRegisterZeroRkeyIndex(int32_t index);
void HccpFakeSetUnregisterResult(int32_t result);
/* Blocks UnregisterMemory inside the library until HccpFakeReleaseUnregister is
 * called. HccpFakeWaitForUnregisterEntry reports whether the call entered the
 * blocked region within timeoutMs. */
void HccpFakeSetUnregisterBlock(int32_t block);
int32_t HccpFakeWaitForUnregisterEntry(uint32_t timeoutMs);
void HccpFakeReleaseUnregister(void);
void HccpFakeSetSendResult(int32_t result);
void HccpFakeSetSendBatchErrorIndex(int32_t index);
void HccpFakeSetAbiVersion(uint32_t version);

uint32_t HccpFakeCreateCalls(void);
uint32_t HccpFakeDeleteCalls(void);
uint32_t HccpFakeRegisterCalls(void);
uint32_t HccpFakeUnregisterCalls(void);
uint32_t HccpFakeSendCalls(void);
uint32_t HccpFakeCapabilitiesCalls(void);
int32_t HccpFakeHandlesPrefilled(void);
uint32_t HccpFakeQpcCount(void);
uint32_t HccpFakeQpcByte(uint32_t qpIndex);
uint32_t HccpFakeSendSlotCount(void);
void* HccpFakeSendSlot(uint32_t index);
uint32_t HccpFakeDeleteManagerCount(void);
void* HccpFakeDeleteManager(uint32_t index);
void* HccpFakeLastManager(void);
void* HccpFakeCapabilitiesManager(void);
uint32_t HccpFakeRegisterDescCount(void);
int32_t HccpFakeRegisterDescMemoryType(uint32_t index);
uint64_t HccpFakeRegisterDescAddr(uint32_t index);
uint64_t HccpFakeRegisterDescSize(uint32_t index);
uint64_t HccpFakeRegisterDescLocalAddr(uint32_t index);
uint32_t HccpFakeUnregisterInfoCount(void);
uint32_t HccpFakeUnregisterInfoRkey(uint32_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KV_SEMANTICS_TESTS_FAKE_HCCP_KV_OVER_ROCE_ABI_H
