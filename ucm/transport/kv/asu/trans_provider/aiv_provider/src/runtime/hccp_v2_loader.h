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
#include <mutex>
#include <sys/types.h>
#include "src/protocol/ub_error_code.h"
#include "src/runtime/hccp_v2_abi.h"

namespace umc::comm::v2 {

using RaInitFunc = int (*)(RaInitConfig*);
using RaDeinitFunc = int (*)(RaInitConfig*);
using TsdProcessOpenFunc = uint32_t (*)(const uint32_t, ProcOpenArgs*);
using TsdProcessCloseFunc = uint32_t (*)(const uint32_t, const pid_t);
using RaGetDevEidInfoNumFunc = int (*)(RaInfo, unsigned int*);
using RaGetDevEidInfoListFunc = int (*)(RaInfo, DevEidInfo[], unsigned int*);
using RaCtxInitFunc = int (*)(CtxInitCfg*, CtxInitAttr*, void**);
using RaGetDevBaseAttrFunc = int (*)(void*, DevBaseAttrT*);
using RaGetAsyncReqResultFunc = int (*)(void*, int*);
using RaCtxChanCreateFunc = int (*)(void*, ChanInfoT*, void**);
using RaCtxCqCreateFunc = int (*)(void*, CqInfoT*, void**);
using RaCtxQpCreateFunc = int (*)(void*, QpCreateAttr*, QpCreateInfo*, void**);
using RaCtxTokenIdAllocFunc = int (*)(void*, HccpTokenId*, void**);
using RaCtxQpImportFunc = int (*)(void*, QpImportInfoT*, void**);
using RaCtxQpBindFunc = int (*)(void*, void*);
using RaCtxLmemRegisterFunc = int (*)(void*, MrRegInfoT*, void**);
using RaCtxRmemImportFunc = int (*)(void*, MrImportInfoT*, void**);
using RaCtxRmemUnimportFunc = int (*)(void*, void*);
using RaCtxLmemUnregisterFunc = int (*)(void*, void*);
using RaCtxQpUnbindFunc = int (*)(void*);
using RaCtxQpUnimportFunc = int (*)(void*, void*);
using RaCtxTokenIdFreeFunc = int (*)(void*, void*);
using RaCtxQpDestroyFunc = int (*)(void*);
using RaCtxCqDestroyFunc = int (*)(void*, void*);
using RaCtxChanDestroyFunc = int (*)(void*, void*);
using RaCtxDeinitFunc = int (*)(void*);
using RaCtxQpQueryBatchFunc = int (*)(void*[], JettyAttr[], unsigned int*);
using RaBatchSendWrFunc = int (*)(void*, SendWrData[], SendWrResp[], unsigned int, unsigned int*);
using RaCtxUpdateCiFunc = int (*)(void*, uint16_t);
using RaCustomChannelFunc = int (*)(RaInfo, CustomChanInfoIn*, CustomChanInfoOut*);
using RaGetTpInfoListAsyncFunc = int (*)(void*, GetTpCfg*, TpInfo[], unsigned int*, void**);
using RaGetIfNumFunc = int (*)(RaGetIfAttr*, unsigned int*);
using RaGetIfAddrsFunc = int (*)(RaGetIfAttr*, InterfaceInfo[], unsigned int*);

using RaSocketInitFunc = int (*)(int, Rdev, void**);
using RaSocketDeinitFunc = int (*)(void*);
using RaSocketBatchConnectFunc = int (*)(SocketConnectInfoT[], unsigned int);
using RaSocketBatchCloseFunc = int (*)(SocketCloseInfoT[], unsigned int);
using RaGetSocketsFunc = int (*)(unsigned int, SocketInfoT[], unsigned int, unsigned int*);
using RaSocketSendFunc = int (*)(const void*, const void*, unsigned long long, unsigned long long*);
using RaSocketRecvFunc = int (*)(const void*, void*, unsigned long long, unsigned long long*);
using RaSocketGetVnicIpInfosFunc = int (*)(unsigned int, IdType, unsigned int[], unsigned int,
                                           IpInfo[]);

class DlHccpV2Api {
public:
    static UbErrorCode LoadLibrary();

    static UbErrorCode CleanUpLibrary();

    static bool IsLoaded();

    static bool SocketApiAvailable();
    static bool SocketVnicIpApiAvailable();

    static uint32_t TsdProcessOpen(const uint32_t logicDeviceId, ProcOpenArgs* args)
    {
        return gTsdProcessOpen(logicDeviceId, args);
    }
    static uint32_t TsdProcessClose(const uint32_t logicDeviceId, const pid_t closePid)
    {
        return gTsdProcessClose(logicDeviceId, closePid);
    }
    static int RaInit(RaInitConfig* cfg) { return gRaInit(cfg); }
    static int RaDeinit(RaInitConfig* cfg) { return gRaDeinit(cfg); }
    static int RaGetDevEidInfoNum(RaInfo info, unsigned int* num)
    {
        return gRaGetDevEidInfoNum(info, num);
    }
    static int RaGetDevEidInfoList(RaInfo info, DevEidInfo list[], unsigned int* num)
    {
        return gRaGetDevEidInfoList(info, list, num);
    }
    static int RaGetIfNum(RaGetIfAttr* cfg, unsigned int* num) { return gRaGetIfNum(cfg, num); }
    static int RaGetIfAddrs(RaGetIfAttr* cfg, InterfaceInfo list[], unsigned int* num)
    {
        return gRaGetIfAddrs(cfg, list, num);
    }

    static int RaSocketInit(int mode, Rdev rdev, void** sock)
    {
        return gRaSocketInit(mode, rdev, sock);
    }
    static int RaSocketDeinit(void* sock) { return gRaSocketDeinit(sock); }
    static int RaSocketBatchConnect(SocketConnectInfoT conn[], unsigned int num)
    {
        return gRaSocketBatchConnect(conn, num);
    }
    static int RaSocketBatchClose(SocketCloseInfoT conn[], unsigned int num)
    {
        return gRaSocketBatchClose(conn, num);
    }
    static int RaGetSockets(unsigned int role, SocketInfoT conn[], unsigned int num,
                            unsigned int* connectedNum)
    {
        return gRaGetSockets(role, conn, num, connectedNum);
    }
    static int RaSocketSend(const void* fd, const void* data, unsigned long long size,
                            unsigned long long* sent)
    {
        return gRaSocketSend(fd, data, size, sent);
    }
    static int RaSocketRecv(const void* fd, void* data, unsigned long long size,
                            unsigned long long* recvd)
    {
        return gRaSocketRecv(fd, data, size, recvd);
    }
    static int RaSocketGetVnicIpInfos(unsigned int phyId, IdType type, unsigned int ids[],
                                      unsigned int num, IpInfo infos[])
    {
        return gRaSocketGetVnicIpInfos(phyId, type, ids, num, infos);
    }

    static int RaCtxInit(CtxInitCfg* cfg, CtxInitAttr* attr, void** ctx)
    {
        return gRaCtxInit(cfg, attr, ctx);
    }
    static int RaGetDevBaseAttr(void* ctx, DevBaseAttrT* attr)
    {
        return gRaGetDevBaseAttr(ctx, attr);
    }
    static int RaCtxTokenIdAlloc(void* ctx, HccpTokenId* info, void** handle)
    {
        return gRaCtxTokenIdAlloc(ctx, info, handle);
    }
    static int RaCtxChanCreate(void* ctx, ChanInfoT* info, void** handle)
    {
        return gRaCtxChanCreate(ctx, info, handle);
    }
    static int RaCtxCqCreate(void* ctx, CqInfoT* info, void** handle)
    {
        return gRaCtxCqCreate(ctx, info, handle);
    }
    static int RaCtxQpCreate(void* ctx, QpCreateAttr* attr, QpCreateInfo* info, void** h)
    {
        return gRaCtxQpCreate(ctx, attr, info, h);
    }
    static int RaCtxLmemRegister(void* ctx, MrRegInfoT* info, void** handle)
    {
        return gRaCtxLmemRegister(ctx, info, handle);
    }
    static int RaCtxQpImport(void* ctx, QpImportInfoT* info, void** handle)
    {
        return gRaCtxQpImport(ctx, info, handle);
    }
    static int RaCtxRmemImport(void* ctx, MrImportInfoT* info, void** handle)
    {
        return gRaCtxRmemImport(ctx, info, handle);
    }
    static int RaCtxQpBind(void* local, void* remote) { return gRaCtxQpBind(local, remote); }
    static int RaGetAsyncReqResult(void* req, int* result)
    {
        return gRaGetAsyncReqResult(req, result);
    }
    static int RaCtxQpQueryBatch(void* handles[], JettyAttr attrs[], unsigned int* num)
    {
        return gRaCtxQpQueryBatch(handles, attrs, num);
    }
    static int RaCustomChannel(RaInfo info, CustomChanInfoIn* in, CustomChanInfoOut* out)
    {
        return gRaCustomChannel(info, in, out);
    }
    static int RaGetTpInfoListAsync(void* ctx, GetTpCfg* cfg, TpInfo list[], unsigned int* num,
                                    void** req)
    {
        return gRaGetTpInfoListAsync(ctx, cfg, list, num, req);
    }

    static int RaBatchSendWr(void* qp, SendWrData wrs[], SendWrResp resps[], unsigned int num,
                             unsigned int* completed)
    {
        return gRaBatchSendWr(qp, wrs, resps, num, completed);
    }
    static int RaCtxUpdateCi(void* qp, uint16_t ci) { return gRaCtxUpdateCi(qp, ci); }

    static int RaCtxQpUnbind(void* qp) { return gRaCtxQpUnbind(qp); }
    static int RaCtxQpUnimport(void* ctx, void* rem) { return gRaCtxQpUnimport(ctx, rem); }
    static int RaCtxRmemUnimport(void* ctx, void* rmem) { return gRaCtxRmemUnimport(ctx, rmem); }
    static int RaCtxLmemUnregister(void* ctx, void* lmem)
    {
        return gRaCtxLmemUnregister(ctx, lmem);
    }
    static int RaCtxQpDestroy(void* qp) { return gRaCtxQpDestroy(qp); }
    static int RaCtxCqDestroy(void* ctx, void* cq) { return gRaCtxCqDestroy(ctx, cq); }
    static int RaCtxChanDestroy(void* ctx, void* chan) { return gRaCtxChanDestroy(ctx, chan); }
    static int RaCtxTokenIdFree(void* ctx, void* token) { return gRaCtxTokenIdFree(ctx, token); }
    static int RaCtxDeinit(void* ctx) { return gRaCtxDeinit(ctx); }

private:
    static void CleanUpLibraryUnlocked();

    static std::mutex gMutex;
    static bool gLoaded;

    static void* gHcclV1LibraryHandler;
    static void* gHcclLibraryHandler;
    static void* gRaLibraryHandler;
    static void* gTsdLibraryHandler;

    static TsdProcessOpenFunc gTsdProcessOpen;
    static TsdProcessCloseFunc gTsdProcessClose;
    static RaInitFunc gRaInit;
    static RaDeinitFunc gRaDeinit;
    static RaGetDevEidInfoNumFunc gRaGetDevEidInfoNum;
    static RaGetDevEidInfoListFunc gRaGetDevEidInfoList;
    static RaGetIfNumFunc gRaGetIfNum;
    static RaGetIfAddrsFunc gRaGetIfAddrs;
    static RaCtxInitFunc gRaCtxInit;
    static RaGetDevBaseAttrFunc gRaGetDevBaseAttr;
    static RaGetAsyncReqResultFunc gRaGetAsyncReqResult;
    static RaCtxChanCreateFunc gRaCtxChanCreate;
    static RaCtxCqCreateFunc gRaCtxCqCreate;
    static RaCtxQpCreateFunc gRaCtxQpCreate;
    static RaCtxTokenIdAllocFunc gRaCtxTokenIdAlloc;
    static RaCtxQpImportFunc gRaCtxQpImport;
    static RaCtxQpBindFunc gRaCtxQpBind;
    static RaCtxLmemRegisterFunc gRaCtxLmemRegister;
    static RaCtxRmemImportFunc gRaCtxRmemImport;
    static RaCtxQpQueryBatchFunc gRaCtxQpQueryBatch;
    static RaCustomChannelFunc gRaCustomChannel;
    static RaGetTpInfoListAsyncFunc gRaGetTpInfoListAsync;
    static RaBatchSendWrFunc gRaBatchSendWr;
    static RaCtxUpdateCiFunc gRaCtxUpdateCi;
    static RaCtxRmemUnimportFunc gRaCtxRmemUnimport;
    static RaCtxLmemUnregisterFunc gRaCtxLmemUnregister;
    static RaCtxQpUnbindFunc gRaCtxQpUnbind;
    static RaCtxQpUnimportFunc gRaCtxQpUnimport;
    static RaCtxTokenIdFreeFunc gRaCtxTokenIdFree;
    static RaCtxQpDestroyFunc gRaCtxQpDestroy;
    static RaCtxCqDestroyFunc gRaCtxCqDestroy;
    static RaCtxChanDestroyFunc gRaCtxChanDestroy;
    static RaCtxDeinitFunc gRaCtxDeinit;
    static RaSocketInitFunc gRaSocketInit;
    static RaSocketDeinitFunc gRaSocketDeinit;
    static RaSocketBatchConnectFunc gRaSocketBatchConnect;
    static RaSocketBatchCloseFunc gRaSocketBatchClose;
    static RaGetSocketsFunc gRaGetSockets;
    static RaSocketSendFunc gRaSocketSend;
    static RaSocketRecvFunc gRaSocketRecv;
    static RaSocketGetVnicIpInfosFunc gRaSocketGetVnicIpInfos;
};

}  // namespace umc::comm::v2
