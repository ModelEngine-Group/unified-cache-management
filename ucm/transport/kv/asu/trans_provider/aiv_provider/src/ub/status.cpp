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

#include "src/ub/status.h"

namespace umc::comm {

const char* UbErrorCodeToString(UbErrorCode code)
{
    switch (code) {
        case UbErrorCode::Ok: return "Ok";

        case UbErrorCode::HccpV2LoadLibraryFailed: return "HccpV2LoadLibraryFailed";
        case UbErrorCode::HccpV2TsdOpenFailed: return "HccpV2TsdOpenFailed";
        case UbErrorCode::HccpV2RaInitFailed: return "HccpV2RaInitFailed";
        case UbErrorCode::HccpV2RaCtxInitFailed: return "HccpV2RaCtxInitFailed";
        case UbErrorCode::HccpV2QpCreateFailed: return "HccpV2QpCreateFailed";
        case UbErrorCode::HccpV2LmemRegisterFailed: return "HccpV2LmemRegisterFailed";
        case UbErrorCode::HccpV2QpImportFailed: return "HccpV2QpImportFailed";
        case UbErrorCode::HccpV2RmemImportFailed: return "HccpV2RmemImportFailed";
        case UbErrorCode::HccpV2QpBindFailed: return "HccpV2QpBindFailed";
        case UbErrorCode::HccpV2CqCreateFailed: return "HccpV2CqCreateFailed";
        case UbErrorCode::HccpV2ChanCreateFailed: return "HccpV2ChanCreateFailed";
        case UbErrorCode::HccpV2TokenIdAllocFailed: return "HccpV2TokenIdAllocFailed";
        case UbErrorCode::HccpV2HandleInvalid: return "HccpV2HandleInvalid";
        case UbErrorCode::HccpV2UboeNotAvailable: return "HccpV2UboeNotAvailable";

        case UbErrorCode::OobConnectFailed: return "OobConnectFailed";
        case UbErrorCode::OobNegotiateFailed: return "OobNegotiateFailed";
        case UbErrorCode::OobStaticYamlNotFound: return "OobStaticYamlNotFound";
        case UbErrorCode::OobStaticYamlMalformed: return "OobStaticYamlMalformed";
        case UbErrorCode::OobCmProtocolMismatch: return "OobCmProtocolMismatch";
        case UbErrorCode::OobCmVersionRejected: return "OobCmVersionRejected";
        case UbErrorCode::OobTransportClosed: return "OobTransportClosed";
        case UbErrorCode::OobTimeout: return "OobTimeout";

        case UbErrorCode::UdmaSqRingFull: return "UdmaSqRingFull";
        case UbErrorCode::UdmaCqPollTimeout: return "UdmaCqPollTimeout";
        case UbErrorCode::UdmaSignalInvalidMagic: return "UdmaSignalInvalidMagic";
        case UbErrorCode::UdmaSignalVersionMismatch: return "UdmaSignalVersionMismatch";
        case UbErrorCode::UdmaQuietTimeout: return "UdmaQuietTimeout";
        case UbErrorCode::UdmaInvalidOpcode: return "UdmaInvalidOpcode";

        case UbErrorCode::KvStripeFull: return "KvStripeFull";
        case UbErrorCode::KvTaskNotFound: return "KvTaskNotFound";
        case UbErrorCode::KvBlockNotFound: return "KvBlockNotFound";
        case UbErrorCode::KvNamespaceInvalid: return "KvNamespaceInvalid";
        case UbErrorCode::KvConnNotFound: return "KvConnNotFound";
        case UbErrorCode::KvMemHandleInvalid: return "KvMemHandleInvalid";
        case UbErrorCode::KvSendBatchEmpty: return "KvSendBatchEmpty";
        case UbErrorCode::KvTaskDescInvalid: return "KvTaskDescInvalid";
        case UbErrorCode::KvTaskChecksumMismatch: return "KvTaskChecksumMismatch";
        case UbErrorCode::KvTaskAddrOutOfRange: return "KvTaskAddrOutOfRange";

        case UbErrorCode::UrmaNotAvailable: return "UrmaNotAvailable";
        case UbErrorCode::UrmaInitFailed: return "UrmaInitFailed";
        case UbErrorCode::UrmaJettyCreateFailed: return "UrmaJettyCreateFailed";
        case UbErrorCode::UrmaSegRegisterFailed: return "UrmaSegRegisterFailed";
        case UbErrorCode::UrmaPostWrFailed: return "UrmaPostWrFailed";
        case UbErrorCode::UrmaPostRecvFailed: return "UrmaPostRecvFailed";
        case UbErrorCode::UrmaPollJfcFailed: return "UrmaPollJfcFailed";
        case UbErrorCode::UrmaImportJettyFailed: return "UrmaImportJettyFailed";
        case UbErrorCode::UrmaReadFailed: return "UrmaReadFailed";
        case UbErrorCode::UrmaWriteFailed: return "UrmaWriteFailed";
        case UbErrorCode::UrmaTpListFailed: return "UrmaTpListFailed";

        case UbErrorCode::InternalAssertionFailed: return "InternalAssertionFailed";
        case UbErrorCode::NotImplemented: return "NotImplemented";
        case UbErrorCode::InvalidArgument: return "InvalidArgument";
    }
    return "UbErrorCode(unknown)";
}

}  // namespace umc::comm
