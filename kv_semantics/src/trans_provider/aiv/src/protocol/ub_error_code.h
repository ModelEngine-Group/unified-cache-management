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

#pragma once

#include <cstdint>

namespace umc::comm {

enum class UbErrorCode : int32_t {
    Ok = 0,

    HccpV2LoadLibraryFailed = 1001,
    HccpV2TsdOpenFailed = 1002,
    HccpV2RaInitFailed = 1003,
    HccpV2RaCtxInitFailed = 1004,
    HccpV2QpCreateFailed = 1005,
    HccpV2LmemRegisterFailed = 1006,
    HccpV2QpImportFailed = 1007,
    HccpV2RmemImportFailed = 1008,
    HccpV2QpBindFailed = 1009,
    HccpV2CqCreateFailed = 1010,
    HccpV2ChanCreateFailed = 1011,
    HccpV2TokenIdAllocFailed = 1012,
    HccpV2HandleInvalid = 1013,
    HccpV2UboeNotAvailable = 1014,

    OobConnectFailed = 2001,
    OobNegotiateFailed = 2002,
    OobStaticYamlNotFound = 2003,
    OobStaticYamlMalformed = 2004,
    OobCmProtocolMismatch = 2005,
    OobCmVersionRejected = 2006,
    OobTransportClosed = 2007,
    OobTimeout = 2008,

    UdmaSqRingFull = 3001,
    UdmaCqPollTimeout = 3002,
    UdmaSignalInvalidMagic = 3003,
    UdmaSignalVersionMismatch = 3004,
    UdmaQuietTimeout = 3005,
    UdmaInvalidOpcode = 3006,

    KvStripeFull = 4001,
    KvTaskNotFound = 4002,
    KvBlockNotFound = 4003,
    KvNamespaceInvalid = 4004,
    KvConnNotFound = 4005,
    KvMemHandleInvalid = 4006,
    KvSendBatchEmpty = 4007,
    KvTaskDescInvalid = 4008,
    KvTaskChecksumMismatch = 4009,
    KvTaskAddrOutOfRange = 4010,

    UrmaNotAvailable = 5001,
    UrmaInitFailed = 5002,
    UrmaJettyCreateFailed = 5003,
    UrmaSegRegisterFailed = 5004,
    UrmaPostWrFailed = 5005,
    UrmaPostRecvFailed = 5006,
    UrmaPollJfcFailed = 5007,
    UrmaImportJettyFailed = 5008,
    UrmaReadFailed = 5009,
    UrmaWriteFailed = 5010,
    UrmaTpListFailed = 5011,

    InternalAssertionFailed = 9001,
    NotImplemented = 9002,
    InvalidArgument = 9003,
};

constexpr int32_t ToInt(UbErrorCode c) { return static_cast<int32_t>(c); }

constexpr UbErrorCode FromInt(int32_t v) { return static_cast<UbErrorCode>(v); }

}  // namespace umc::comm

#ifdef __CCE__
#define UB_ERR_OK 0
#define UB_ERR_UDMA_SQ_RING_FULL 3001
#define UB_ERR_UDMA_CQ_POLL_TIMEOUT 3002
#define UB_ERR_UDMA_SIGNAL_INVALID_MAGIC 3003
#define UB_ERR_UDMA_SIGNAL_VERSION_MISMATCH 3004
#define UB_ERR_UDMA_QUIET_TIMEOUT 3005
#define UB_ERR_UDMA_INVALID_OPCODE 3006
#endif
