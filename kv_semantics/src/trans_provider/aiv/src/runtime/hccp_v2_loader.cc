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

#include "src/runtime/hccp_v2_loader.h"
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include "src/ub/log.h"

namespace umc::comm::v2 {

std::mutex DlHccpV2Api::gMutex;
bool DlHccpV2Api::gLoaded = false;

void* DlHccpV2Api::gHcclV1LibraryHandler = nullptr;
void* DlHccpV2Api::gHcclLibraryHandler = nullptr;
void* DlHccpV2Api::gRaLibraryHandler = nullptr;
void* DlHccpV2Api::gTsdLibraryHandler = nullptr;

TsdProcessOpenFunc DlHccpV2Api::gTsdProcessOpen = nullptr;
TsdProcessCloseFunc DlHccpV2Api::gTsdProcessClose = nullptr;
RaInitFunc DlHccpV2Api::gRaInit = nullptr;
RaDeinitFunc DlHccpV2Api::gRaDeinit = nullptr;
RaGetDevEidInfoNumFunc DlHccpV2Api::gRaGetDevEidInfoNum = nullptr;
RaGetDevEidInfoListFunc DlHccpV2Api::gRaGetDevEidInfoList = nullptr;
RaGetIfNumFunc DlHccpV2Api::gRaGetIfNum = nullptr;
RaGetIfAddrsFunc DlHccpV2Api::gRaGetIfAddrs = nullptr;
RaCtxInitFunc DlHccpV2Api::gRaCtxInit = nullptr;
RaGetDevBaseAttrFunc DlHccpV2Api::gRaGetDevBaseAttr = nullptr;
RaGetAsyncReqResultFunc DlHccpV2Api::gRaGetAsyncReqResult = nullptr;
RaCtxChanCreateFunc DlHccpV2Api::gRaCtxChanCreate = nullptr;
RaCtxCqCreateFunc DlHccpV2Api::gRaCtxCqCreate = nullptr;
RaCtxQpCreateFunc DlHccpV2Api::gRaCtxQpCreate = nullptr;
RaCtxTokenIdAllocFunc DlHccpV2Api::gRaCtxTokenIdAlloc = nullptr;
RaCtxQpImportFunc DlHccpV2Api::gRaCtxQpImport = nullptr;
RaCtxQpBindFunc DlHccpV2Api::gRaCtxQpBind = nullptr;
RaCtxLmemRegisterFunc DlHccpV2Api::gRaCtxLmemRegister = nullptr;
RaCtxRmemImportFunc DlHccpV2Api::gRaCtxRmemImport = nullptr;
RaCtxQpQueryBatchFunc DlHccpV2Api::gRaCtxQpQueryBatch = nullptr;
RaCustomChannelFunc DlHccpV2Api::gRaCustomChannel = nullptr;
RaGetTpInfoListAsyncFunc DlHccpV2Api::gRaGetTpInfoListAsync = nullptr;
RaBatchSendWrFunc DlHccpV2Api::gRaBatchSendWr = nullptr;
RaCtxUpdateCiFunc DlHccpV2Api::gRaCtxUpdateCi = nullptr;
RaCtxRmemUnimportFunc DlHccpV2Api::gRaCtxRmemUnimport = nullptr;
RaCtxLmemUnregisterFunc DlHccpV2Api::gRaCtxLmemUnregister = nullptr;
RaCtxQpUnbindFunc DlHccpV2Api::gRaCtxQpUnbind = nullptr;
RaCtxQpUnimportFunc DlHccpV2Api::gRaCtxQpUnimport = nullptr;
RaCtxTokenIdFreeFunc DlHccpV2Api::gRaCtxTokenIdFree = nullptr;
RaCtxQpDestroyFunc DlHccpV2Api::gRaCtxQpDestroy = nullptr;
RaCtxCqDestroyFunc DlHccpV2Api::gRaCtxCqDestroy = nullptr;
RaCtxChanDestroyFunc DlHccpV2Api::gRaCtxChanDestroy = nullptr;
RaCtxDeinitFunc DlHccpV2Api::gRaCtxDeinit = nullptr;
RaSocketInitFunc DlHccpV2Api::gRaSocketInit = nullptr;
RaSocketDeinitFunc DlHccpV2Api::gRaSocketDeinit = nullptr;
RaSocketBatchConnectFunc DlHccpV2Api::gRaSocketBatchConnect = nullptr;
RaSocketBatchCloseFunc DlHccpV2Api::gRaSocketBatchClose = nullptr;
RaGetSocketsFunc DlHccpV2Api::gRaGetSockets = nullptr;
RaSocketSendFunc DlHccpV2Api::gRaSocketSend = nullptr;
RaSocketRecvFunc DlHccpV2Api::gRaSocketRecv = nullptr;
RaSocketGetVnicIpInfosFunc DlHccpV2Api::gRaSocketGetVnicIpInfos = nullptr;

namespace {

template <class Func>
bool LoadSymAlt(Func& slot, void* handle, const char* primary, const char* fallback)
{
    dlerror();
    void* sym = dlsym(handle, primary);
    if (sym == nullptr) { sym = dlsym(handle, fallback); }
    if (sym == nullptr) {
        const char* error = dlerror();
        UB_LOG_ERROR("dlsym failed for both '{}' and '{}': {}", primary, fallback,
                     error != nullptr ? error : "(null)");
        slot = nullptr;
        return false;
    }
    slot = reinterpret_cast<Func>(sym);
    return true;
}

template <class Func>
bool LoadOptionalSymMulti(Func& slot, std::initializer_list<void*> handles, const char* primary,
                          const char* fallback)
{
    for (void* h : handles) {
        if (h == nullptr) continue;
        dlerror();
        void* sym = dlsym(h, primary);
        if (sym == nullptr) sym = dlsym(h, fallback);
        if (sym != nullptr) {
            slot = reinterpret_cast<Func>(sym);
            return true;
        }
    }
    slot = nullptr;
    return false;
}

void* DlOpenOrLog(const char* name)
{
    void* handle = dlopen(name, RTLD_NOW);
    if (handle == nullptr) {
        UB_LOG_ERROR(
            "dlopen failed for '{}': {}. Source scripts/env.sh, or add Ascend "
            "driver lib path to LD_LIBRARY_PATH.",
            name, dlerror());
    }
    return handle;
}

}  // namespace

UbErrorCode DlHccpV2Api::LoadLibrary()
{
    std::lock_guard<std::mutex> lock(gMutex);
    if (gLoaded) return UbErrorCode::Ok;

    gHcclV1LibraryHandler = DlOpenOrLog("libhccl.so");
    if (!gHcclV1LibraryHandler) return UbErrorCode::HccpV2LoadLibraryFailed;

    gHcclLibraryHandler = DlOpenOrLog("libhccl_v2.so");
    if (!gHcclLibraryHandler) {
        dlclose(gHcclV1LibraryHandler);
        gHcclV1LibraryHandler = nullptr;
        return UbErrorCode::HccpV2LoadLibraryFailed;
    }

    gRaLibraryHandler = DlOpenOrLog("libra.so");
    if (!gRaLibraryHandler) {
        dlclose(gHcclLibraryHandler);
        gHcclLibraryHandler = nullptr;
        dlclose(gHcclV1LibraryHandler);
        gHcclV1LibraryHandler = nullptr;
        return UbErrorCode::HccpV2LoadLibraryFailed;
    }

    gTsdLibraryHandler = DlOpenOrLog("libtsdclient.so");
    if (!gTsdLibraryHandler) {
        dlclose(gRaLibraryHandler);
        gRaLibraryHandler = nullptr;
        dlclose(gHcclLibraryHandler);
        gHcclLibraryHandler = nullptr;
        dlclose(gHcclV1LibraryHandler);
        gHcclV1LibraryHandler = nullptr;
        return UbErrorCode::HccpV2LoadLibraryFailed;
    }

    bool ok = true;
    ok &= LoadSymAlt(gRaInit, gHcclLibraryHandler, "RaInit", "ra_init");
    ok &= LoadSymAlt(gRaDeinit, gHcclLibraryHandler, "RaDeinit", "ra_deinit");
    ok &= LoadSymAlt(gRaGetDevEidInfoNum, gHcclLibraryHandler, "RaGetDevEidInfoNum",
                     "ra_get_dev_eid_info_num");
    ok &= LoadSymAlt(gRaGetDevEidInfoList, gHcclLibraryHandler, "RaGetDevEidInfoList",
                     "ra_get_dev_eid_info_list");
    ok &= LoadSymAlt(gRaCtxInit, gHcclLibraryHandler, "RaCtxInit", "ra_ctx_init");
    ok &= LoadSymAlt(gRaGetDevBaseAttr, gHcclLibraryHandler, "RaGetDevBaseAttr",
                     "ra_get_dev_base_attr");
    ok &= LoadSymAlt(gRaGetAsyncReqResult, gHcclLibraryHandler, "RaGetAsyncReqResult",
                     "ra_get_async_req_result");
    ok &= LoadSymAlt(gRaCtxCqCreate, gHcclLibraryHandler, "RaCtxCqCreate", "ra_ctx_cq_create");
    ok &= LoadSymAlt(gRaCtxQpCreate, gHcclLibraryHandler, "RaCtxQpCreate", "ra_ctx_qp_create");
    ok &= LoadSymAlt(gRaCtxTokenIdAlloc, gHcclLibraryHandler, "RaCtxTokenIdAlloc",
                     "ra_ctx_token_id_alloc");
    ok &= LoadSymAlt(gRaCtxQpImport, gHcclLibraryHandler, "RaCtxQpImport", "ra_ctx_qp_import");
    ok &= LoadSymAlt(gRaCtxQpBind, gHcclLibraryHandler, "RaCtxQpBind", "ra_ctx_qp_bind");
    ok &= LoadSymAlt(gRaCtxLmemRegister, gHcclLibraryHandler, "RaCtxLmemRegister",
                     "ra_ctx_lmem_register");
    ok &=
        LoadSymAlt(gRaCtxRmemImport, gHcclLibraryHandler, "RaCtxRmemImport", "ra_ctx_rmem_import");
    ok &= LoadSymAlt(gRaCtxRmemUnimport, gHcclLibraryHandler, "RaCtxRmemUnimport",
                     "ra_ctx_rmem_unimport");
    ok &= LoadSymAlt(gRaCtxLmemUnregister, gHcclLibraryHandler, "RaCtxLmemUnregister",
                     "ra_ctx_lmem_unregister");
    ok &= LoadSymAlt(gRaCtxQpUnbind, gHcclLibraryHandler, "RaCtxQpUnbind", "ra_ctx_qp_unbind");
    ok &=
        LoadSymAlt(gRaCtxQpUnimport, gHcclLibraryHandler, "RaCtxQpUnimport", "ra_ctx_qp_unimport");
    ok &= LoadSymAlt(gRaCtxTokenIdFree, gHcclLibraryHandler, "RaCtxTokenIdFree",
                     "ra_ctx_token_id_free");
    ok &= LoadSymAlt(gRaCtxQpDestroy, gHcclLibraryHandler, "RaCtxQpDestroy", "ra_ctx_qp_destroy");
    ok &= LoadSymAlt(gRaCtxCqDestroy, gHcclLibraryHandler, "RaCtxCqDestroy", "ra_ctx_cq_destroy");
    ok &= LoadSymAlt(gRaCtxDeinit, gHcclLibraryHandler, "RaCtxDeinit", "ra_ctx_deinit");
    ok &= LoadSymAlt(gRaBatchSendWr, gHcclLibraryHandler, "RaBatchSendWr", "ra_batch_send_wr");
    ok &= LoadSymAlt(gRaCtxUpdateCi, gHcclLibraryHandler, "RaCtxUpdateCi", "ra_ctx_update_ci");
    ok &= LoadSymAlt(gRaCustomChannel, gHcclLibraryHandler, "RaCustomChannel", "ra_custom_channel");
    ok &= LoadSymAlt(gRaGetTpInfoListAsync, gHcclLibraryHandler, "RaGetTpInfoListAsync",
                     "ra_get_tp_info_list_async");

    ok &= LoadSymAlt(gRaCtxChanCreate, gRaLibraryHandler, "RaCtxChanCreate", "ra_ctx_chan_create");
    ok &=
        LoadSymAlt(gRaCtxChanDestroy, gRaLibraryHandler, "RaCtxChanDestroy", "ra_ctx_chan_destroy");
    ok &= LoadSymAlt(gRaCtxQpQueryBatch, gRaLibraryHandler, "RaCtxQpQueryBatch",
                     "ra_ctx_qp_query_batch");
    ok &= LoadSymAlt(gRaGetIfNum, gRaLibraryHandler, "RaGetIfnum", "ra_get_if_num");
    ok &= LoadSymAlt(gRaGetIfAddrs, gRaLibraryHandler, "RaGetIfaddrs", "ra_get_if_addrs");

    ok &= LoadSymAlt(gTsdProcessOpen, gTsdLibraryHandler, "TsdProcessOpen", "tsd_process_open");
    ok &= LoadSymAlt(gTsdProcessClose, gTsdLibraryHandler, "TsdProcessClose", "tsd_process_close");

    if (!ok) {
        UB_LOG_ERROR(
            "DlHccpV2Api::LoadLibrary failed: one or more dlsym lookups returned null. "
            "CANN 9.0 may be required; current build will not run on this machine.");
        CleanUpLibraryUnlocked();
        return UbErrorCode::HccpV2LoadLibraryFailed;
    }

    const std::initializer_list<void*> socketHandles = {gHcclLibraryHandler, gRaLibraryHandler};
    bool sockOk = true;
    sockOk &= LoadOptionalSymMulti(gRaSocketInit, socketHandles, "RaSocketInit", "ra_socket_init");
    sockOk &=
        LoadOptionalSymMulti(gRaSocketDeinit, socketHandles, "RaSocketDeinit", "ra_socket_deinit");
    sockOk &= LoadOptionalSymMulti(gRaSocketBatchConnect, socketHandles, "RaSocketBatchConnect",
                                   "ra_socket_batch_connect");
    sockOk &= LoadOptionalSymMulti(gRaSocketBatchClose, socketHandles, "RaSocketBatchClose",
                                   "ra_socket_batch_close");
    sockOk &= LoadOptionalSymMulti(gRaGetSockets, socketHandles, "RaGetSockets", "ra_get_sockets");
    sockOk &= LoadOptionalSymMulti(gRaSocketSend, socketHandles, "RaSocketSend", "ra_socket_send");
    sockOk &= LoadOptionalSymMulti(gRaSocketRecv, socketHandles, "RaSocketRecv", "ra_socket_recv");
    const bool vnicIpOk =
        LoadOptionalSymMulti(gRaSocketGetVnicIpInfos, socketHandles, "RaSocketGetVnicIpInfos",
                             "ra_socket_get_vnic_ip_infos");

    gLoaded = true;
    UB_LOG_DEBUG("DlHccpV2Api: loaded 4 SO + 32 core symbols + RaSocket={} + VnicIp={}",
                 sockOk ? "available" : "unavailable(fallback to TCP/StaticConfig)",
                 vnicIpOk ? "available" : "unavailable(fallback to RaGetIfaddrs)");
    return UbErrorCode::Ok;
}

UbErrorCode DlHccpV2Api::CleanUpLibrary()
{
    // Calls use resolved function pointers without a per-call lock. Keep the
    // successfully loaded libraries resident for the process lifetime. The
    // unlocked helper remains only for a failed LoadLibrary rollback.
    return UbErrorCode::Ok;
}

void DlHccpV2Api::CleanUpLibraryUnlocked()
{
    if (!gLoaded && !gHcclV1LibraryHandler && !gHcclLibraryHandler && !gRaLibraryHandler &&
        !gTsdLibraryHandler) {
        return;
    }

    gTsdProcessOpen = nullptr;
    gTsdProcessClose = nullptr;
    gRaInit = nullptr;
    gRaDeinit = nullptr;
    gRaGetDevEidInfoNum = nullptr;
    gRaGetDevEidInfoList = nullptr;
    gRaGetIfNum = nullptr;
    gRaGetIfAddrs = nullptr;
    gRaCtxInit = nullptr;
    gRaGetDevBaseAttr = nullptr;
    gRaGetAsyncReqResult = nullptr;
    gRaCtxChanCreate = nullptr;
    gRaCtxCqCreate = nullptr;
    gRaCtxQpCreate = nullptr;
    gRaCtxTokenIdAlloc = nullptr;
    gRaCtxQpImport = nullptr;
    gRaCtxQpBind = nullptr;
    gRaCtxLmemRegister = nullptr;
    gRaCtxRmemImport = nullptr;
    gRaCtxQpQueryBatch = nullptr;
    gRaCustomChannel = nullptr;
    gRaGetTpInfoListAsync = nullptr;
    gRaBatchSendWr = nullptr;
    gRaCtxUpdateCi = nullptr;
    gRaCtxRmemUnimport = nullptr;
    gRaCtxLmemUnregister = nullptr;
    gRaCtxQpUnbind = nullptr;
    gRaCtxQpUnimport = nullptr;
    gRaCtxTokenIdFree = nullptr;
    gRaCtxQpDestroy = nullptr;
    gRaCtxCqDestroy = nullptr;
    gRaCtxChanDestroy = nullptr;
    gRaCtxDeinit = nullptr;
    gRaSocketInit = nullptr;
    gRaSocketDeinit = nullptr;
    gRaSocketBatchConnect = nullptr;
    gRaSocketBatchClose = nullptr;
    gRaGetSockets = nullptr;
    gRaSocketSend = nullptr;
    gRaSocketRecv = nullptr;
    gRaSocketGetVnicIpInfos = nullptr;

    auto close_one = [](void*& h, const char* name) {
        if (h) {
            if (dlclose(h) != 0) { UB_LOG_WARN("dlclose({}) failed: {}", name, dlerror()); }
            h = nullptr;
        }
    };

    close_one(gTsdLibraryHandler, "libtsdclient.so");
    close_one(gRaLibraryHandler, "libra.so");
    close_one(gHcclLibraryHandler, "libhccl_v2.so");
    close_one(gHcclV1LibraryHandler, "libhccl.so");

    gLoaded = false;
}

bool DlHccpV2Api::IsLoaded()
{
    std::lock_guard<std::mutex> lock(gMutex);
    return gLoaded;
}

bool DlHccpV2Api::SocketApiAvailable()
{
    std::lock_guard<std::mutex> lock(gMutex);
    return gLoaded && gRaSocketInit && gRaSocketDeinit && gRaSocketBatchConnect &&
           gRaSocketBatchClose && gRaGetSockets && gRaSocketSend && gRaSocketRecv;
}

bool DlHccpV2Api::SocketVnicIpApiAvailable()
{
    std::lock_guard<std::mutex> lock(gMutex);
    return gLoaded && gRaSocketGetVnicIpInfos;
}

}  // namespace umc::comm::v2
