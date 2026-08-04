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

#include <cstddef>
#include <cstdint>

namespace umc::comm {

enum class UdmaDbMode : int32_t {
    Invalid = -1,
    HwDb = 0,
    SwDb = 1,
};

enum class UdmaOpcode : uint32_t {
    Send = 0,
    SendWithImm = 1,
    SendWithInv = 2,
    Write = 3,
    WriteWithImm = 4,
    WriteWithNotify = 5,
    Read = 6,
    Cas = 7,
    AtomicSwap = 8,
    AtomicStore = 9,
    AtomicLoad = 0xA,
    Faa = 0xB,
    Nop = 0x11,
};

#pragma pack(push, 1)
struct UdmaEid {
    uint8_t raw[16];
};
#pragma pack(pop)
static_assert(sizeof(UdmaEid) == 16, "UdmaEid must be 16 bytes");

#pragma pack(push, 8)
struct UdmaWqCtx {
    uint32_t wqn;           // [0,4)
    uint64_t bufAddr;       // [8,16)
    uint32_t wqeShiftSize;  // [16,20)  log2(wqebbSize)
    uint32_t depth;         // [20,24)  ring depth
    uint64_t headAddr;      // [24,32)  Producer Index addr
    uint64_t tailAddr;      // [32,40)  Consumer Index addr
    UdmaDbMode dbMode;      // [40,44)
    uint64_t dbAddr;        // [48,56)  doorbell MMIO
    uint32_t sl;            // [56,60)  service level
    uint64_t wqeCntAddr;    // [64,72)  wqe count counter
    uint64_t amoAddr;       // [72,80)  atomic fetch data area
};
#pragma pack(pop)
static_assert(sizeof(UdmaWqCtx) == 80, "UdmaWqCtx must be 80 bytes");
static_assert(offsetof(UdmaWqCtx, wqn) == 0, "");
static_assert(offsetof(UdmaWqCtx, bufAddr) == 8, "");
static_assert(offsetof(UdmaWqCtx, wqeShiftSize) == 16, "");
static_assert(offsetof(UdmaWqCtx, depth) == 20, "");
static_assert(offsetof(UdmaWqCtx, headAddr) == 24, "");
static_assert(offsetof(UdmaWqCtx, tailAddr) == 32, "");
static_assert(offsetof(UdmaWqCtx, dbMode) == 40, "");
static_assert(offsetof(UdmaWqCtx, dbAddr) == 48, "");
static_assert(offsetof(UdmaWqCtx, sl) == 56, "");
static_assert(offsetof(UdmaWqCtx, wqeCntAddr) == 64, "");
static_assert(offsetof(UdmaWqCtx, amoAddr) == 72, "");

#pragma pack(push, 8)
struct UdmaCqCtx {
    uint32_t cqn;           // [0,4)
    uint64_t bufAddr;       // [8,16)
    uint32_t cqeShiftSize;  // [16,20)  log2(cqeSize)
    uint32_t depth;         // [20,24)
    uint64_t headAddr;      // [24,32)
    uint64_t tailAddr;      // [32,40)
    UdmaDbMode dbMode;      // [40,44)
    uint64_t dbAddr;        // [48,56)
};
#pragma pack(pop)
static_assert(sizeof(UdmaCqCtx) == 56, "UdmaCqCtx must be 56 bytes");
static_assert(offsetof(UdmaCqCtx, cqn) == 0, "");
static_assert(offsetof(UdmaCqCtx, bufAddr) == 8, "");
static_assert(offsetof(UdmaCqCtx, cqeShiftSize) == 16, "");
static_assert(offsetof(UdmaCqCtx, depth) == 20, "");
static_assert(offsetof(UdmaCqCtx, headAddr) == 24, "");
static_assert(offsetof(UdmaCqCtx, tailAddr) == 32, "");
static_assert(offsetof(UdmaCqCtx, dbMode) == 40, "");
static_assert(offsetof(UdmaCqCtx, dbAddr) == 48, "");

#pragma pack(push, 8)
struct UdmaSegInfo {
    uint8_t tokenValueValid;  // [0,1)
    uint8_t rmtJettyType;     // [1,2)
    uint8_t targetHint;       // [2,3)
    uint8_t _pad0;            // [3,4)
    uint32_t tpn;             // [4,8)
    uint32_t tid;             // [8,12)
    uint32_t rmtTokenValue;   // [12,16)
    uint64_t len;             // [16,24)
    uint64_t addr;            // [24,32)  seg base addr
    uint64_t eidAddr;         // [32,40)
};
#pragma pack(pop)
static_assert(sizeof(UdmaSegInfo) == 40, "UdmaSegInfo must be 40 bytes");
static_assert(offsetof(UdmaSegInfo, tokenValueValid) == 0, "");
static_assert(offsetof(UdmaSegInfo, rmtJettyType) == 1, "");
static_assert(offsetof(UdmaSegInfo, targetHint) == 2, "");
static_assert(offsetof(UdmaSegInfo, tpn) == 4, "");
static_assert(offsetof(UdmaSegInfo, tid) == 8, "");
static_assert(offsetof(UdmaSegInfo, rmtTokenValue) == 12, "");
static_assert(offsetof(UdmaSegInfo, len) == 16, "");
static_assert(offsetof(UdmaSegInfo, addr) == 24, "");
static_assert(offsetof(UdmaSegInfo, eidAddr) == 32, "");

struct UdmaSqeCtx {
    uint32_t sqeBbIdx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    uint32_t targetHint : 8;
    uint32_t opcode : 8;  // UdmaOpcode
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    uint32_t tpId : 24;
    uint32_t sgeNum : 8;
    uint32_t rmtJettyOrSegId : 20;
    uint32_t rsv2 : 12;
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    uint32_t rmtTokenValue;
    uint32_t udfType : 8;
    uint32_t reduceDataType : 4;
    uint32_t reduceOpcode : 4;
    uint32_t rsv3 : 16;
    uint32_t rmtAddrLOrTokenId;
    uint32_t rmtAddrHOrTokenValue;
};
static_assert(sizeof(UdmaSqeCtx) == 48, "UdmaSqeCtx must be 48 bytes (12 words)");
static_assert(alignof(UdmaSqeCtx) == 8, "UdmaSqeCtx natural alignment must be 8 (no pack)");
static_assert(offsetof(UdmaSqeCtx, rmtEidL) == 16, "");
static_assert(offsetof(UdmaSqeCtx, rmtEidH) == 24, "");
static_assert(offsetof(UdmaSqeCtx, rmtTokenValue) == 32, "");
static_assert(offsetof(UdmaSqeCtx, rmtAddrLOrTokenId) == 40, "");
static_assert(offsetof(UdmaSqeCtx, rmtAddrHOrTokenValue) == 44, "");

struct UdmaAivInfo {
    uint32_t qpNum;          // [0,4)
    uint32_t peerCount;      // [4,8)   peer slots, including self
    uint64_t sqPtr;          // [8,16)  UdmaWqCtx[peerCount * qpNum]
    uint64_t rqPtr;          // [16,24) UdmaWqCtx[peerCount * qpNum]
    uint64_t scqPtr;         // [24,32) UdmaCqCtx[peerCount * qpNum]
    uint64_t rcqPtr;         // [32,40) UdmaCqCtx[...]
    uint64_t memPtr;         // [40,48) UdmaSegInfo[peerCount * qpNum]
    uint64_t signalSlotPtr;  // [48,56) UbSignalSlot[peerCount * stripeCount]
    uint64_t flagSlotPtr;    // [56,64) completion flags
};
static_assert(sizeof(UdmaAivInfo) == 64, "UdmaAivInfo must be 64 bytes");
static_assert(offsetof(UdmaAivInfo, qpNum) == 0, "");
static_assert(offsetof(UdmaAivInfo, peerCount) == 4, "");
static_assert(offsetof(UdmaAivInfo, sqPtr) == 8, "");
static_assert(offsetof(UdmaAivInfo, rqPtr) == 16, "");
static_assert(offsetof(UdmaAivInfo, scqPtr) == 24, "");
static_assert(offsetof(UdmaAivInfo, rcqPtr) == 32, "");
static_assert(offsetof(UdmaAivInfo, memPtr) == 40, "");
static_assert(offsetof(UdmaAivInfo, signalSlotPtr) == 48, "");
static_assert(offsetof(UdmaAivInfo, flagSlotPtr) == 56, "");

}  // namespace umc::comm
