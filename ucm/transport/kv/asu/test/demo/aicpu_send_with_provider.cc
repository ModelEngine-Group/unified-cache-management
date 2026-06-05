/*
 * AICPU Send example using AICPUTransProvider.
 *
 * Flow (same as hixl demo):
 *   InitEndpoint -> RegisterMemory -> ExportMemory -> WritePeerFile
 *   -> ReadPeerFile -> ImportMemory -> CreateConnection -> AllocThread
 *   -> build sendRecvContext -> LaunchKernel
 *
 * Run two processes:
 *   Rank 0: ./aicpu_send_with_provider --rank=0 --logic-dev=0 --phy-dev=0 --ip=192.168.190.170 --bytes=4096 \
 *             --local-file=/tmp/r0.bin --peer-file=/tmp/r1.bin --done-file=/tmp/hixl.done \
 *             --kernel-json=./libcann_hixl_kernel.json
 *
 *   Rank 1: ./aicpu_send_with_provider --rank=1 --logic-dev=2 --phy-dev=2 --ip=192.168.190.172 --bytes=4096 \
 *             --local-file=/tmp/r1.bin --peer-file=/tmp/r0.bin --done-file=/tmp/hixl.done \
 *             --kernel-json=./libcann_hixl_kernel.json
 */

#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hixl_kernel/hixl_send.h"
#include "hcomm/hcomm_primitives.h"
#include "hcomm/hcomm_res_defs.h"
#include "aicpu_trans_provider.h"

namespace {

constexpr uint32_t kMagic = 0x48585344;
constexpr uint32_t kVersion = 1;
constexpr uint16_t kDefaultPort = 16666;
constexpr uint32_t kMailboxMagic = 0x4853524d;

struct PeerFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t descLen;
    EndpointDesc endpoint;
};

struct Options {
    int rank = -1;
    int logicDev = -1;
    uint32_t phyDev = 0;
    uint64_t bytes = 4096;
    std::string ip;
    std::string localFile;
    std::string peerFile;
    std::string doneFile;
    std::string kernelJson;
    std::string ipMap;
    std::string message = "Hello world";
};

void Log(const Options &opt, const char *fmt, ...)
{
    printf("[rank %d] ", opt.rank);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

int Fail(const char *what, int ret, int line)
{
    fprintf(stderr, "FAIL line %d: %s ret=%d\n", line, what, ret);
    return ret == 0 ? 1 : ret;
}

void PrintBytes(const Options &opt, const char *label, const uint8_t *data, size_t len)
{
    const size_t shown = len < 64U ? len : 64U;
    printf("[rank %d] %s first %zu/%zu bytes:", opt.rank, label, shown, len);
    for (size_t i = 0; i < shown; ++i) {
        printf(" %02x", static_cast<unsigned int>(data[i]));
    }
    printf("\n");
    fflush(stdout);
}

#define CHECK_RET(expr) do { int _ret = static_cast<int>(expr); if (_ret != 0) return Fail(#expr, _ret, __LINE__); } while (0)
#define CHECK_ACL(expr) do { aclError _ret = (expr); if (_ret != ACL_SUCCESS) return Fail(#expr, static_cast<int>(_ret), __LINE__); } while (0)
#define CHECK_STATUS(expr) do { auto _s = (expr); if (!_s.ok()) { fprintf(stderr, "FAIL: %s: %s\n", #expr, _s.message.c_str()); return 1; } } while (0)
#define CHECK_VEC_STATUS(expr) do { auto _vs = (expr); for (const auto& _s : _vs) { if (!_s.ok()) { fprintf(stderr, "FAIL: %s: %s\n", #expr, _s.message.c_str()); return 1; } } } while (0)

bool ParseIntArg(const char *arg, const char *name, int *out)
{
    const size_t n = strlen(name);
    if (strncmp(arg, name, n) != 0 || arg[n] != '=') {
        return false;
    }
    *out = atoi(arg + n + 1);
    return true;
}

bool ParseU64Arg(const char *arg, const char *name, uint64_t *out)
{
    const size_t n = strlen(name);
    if (strncmp(arg, name, n) != 0 || arg[n] != '=') {
        return false;
    }
    *out = strtoull(arg + n + 1, nullptr, 0);
    return true;
}

bool ParseStrArg(const char *arg, const char *name, std::string *out)
{
    const size_t n = strlen(name);
    if (strncmp(arg, name, n) != 0 || arg[n] != '=') {
        return false;
    }
    *out = arg + n + 1;
    return true;
}

void Usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s --rank=0|1 --logic-dev=N --phy-dev=N --ip=A.B.C.D --bytes=N \\\n"
        "          --local-file=/tmp/r0.bin --peer-file=/tmp/r1.bin \\\n"
        "          --done-file=/tmp/hixl.done --kernel-json=/path/libcann_hixl_kernel.json \\\n"
        "          [--ip-map=/tmp/npu_ip_map.txt] [--message='Hello world']\n",
        prog);
}

bool ParseOptions(int argc, char **argv, Options *opt)
{
    for (int i = 1; i < argc; ++i) {
        int tmp = 0;
        uint64_t u64 = 0;
        if (ParseIntArg(argv[i], "--rank", &tmp)) {
            opt->rank = tmp;
        } else if (ParseIntArg(argv[i], "--logic-dev", &tmp)) {
            opt->logicDev = tmp;
        } else if (ParseIntArg(argv[i], "--phy-dev", &tmp)) {
            opt->phyDev = static_cast<uint32_t>(tmp);
        } else if (ParseU64Arg(argv[i], "--bytes", &u64)) {
            opt->bytes = u64;
        } else if (ParseStrArg(argv[i], "--ip", &opt->ip)) {
        } else if (ParseStrArg(argv[i], "--local-file", &opt->localFile)) {
        } else if (ParseStrArg(argv[i], "--peer-file", &opt->peerFile)) {
        } else if (ParseStrArg(argv[i], "--done-file", &opt->doneFile)) {
        } else if (ParseStrArg(argv[i], "--kernel-json", &opt->kernelJson)) {
        } else if (ParseStrArg(argv[i], "--ip-map", &opt->ipMap)) {
        } else if (ParseStrArg(argv[i], "--message", &opt->message)) {
        } else {
            return false;
        }
    }

    return (opt->rank == 0 || opt->rank == 1) && opt->logicDev >= 0 && 
           !opt->ip.empty() &&
           !opt->localFile.empty() && !opt->peerFile.empty() && 
           !opt->doneFile.empty() && !opt->kernelJson.empty() &&
           !opt->message.empty();
}



int WritePeerFile(const std::string &path, const EndpointDesc &ep,
                  const void *desc, uint32_t descLen)
{
    PeerFileHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.descLen = descLen;
    hdr.endpoint = ep;

    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        perror(path.c_str());
        return 1;
    }
    os.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    os.write(static_cast<const char *>(desc), descLen);
    return os.good() ? 0 : 1;
}

int ReadPeerFile(const std::string &path, EndpointDesc *ep, std::vector<char> *desc)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        return 1;
    }
    PeerFileHeader hdr{};
    is.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!is.good() || hdr.magic != kMagic || hdr.version != kVersion || hdr.descLen == 0) {
        return 1;
    }
    desc->resize(hdr.descLen);
    is.read(desc->data(), hdr.descLen);
    if (!is.good()) {
        return 1;
    }
    *ep = hdr.endpoint;
    return 0;
}

int WaitForPeerFile(const Options &opt, const std::string &path,
                    EndpointDesc *ep, std::vector<char> *desc)
{
    uint32_t attempts = 0;
    for (;;) {
        if (ReadPeerFile(path, ep, desc) == 0) {
            return 0;
        }
        ++attempts;
        if (attempts % 50 == 0) {
            Log(opt, "still waiting for peer descriptor after %u ms", attempts * 100);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int LoadKernel(const std::string &json, const char *funcName, aclrtBinHandle *bin, aclrtFuncHandle *func)
{
    aclrtBinaryLoadOptions loadOptions{};
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    CHECK_ACL(aclrtBinaryLoadFromFile(json.c_str(), &loadOptions, bin));
    CHECK_ACL(aclrtBinaryGetFunction(*bin, funcName, func));
    return 0;
}

int LaunchKernel(aclrtFuncHandle func, aclrtStream stream, void *args, size_t argsSize)
{
    aclrtArgsHandle argsHandle = nullptr;
    aclrtParamHandle paramHandle = nullptr;
    CHECK_ACL(aclrtKernelArgsInit(func, &argsHandle));
    CHECK_ACL(aclrtKernelArgsAppend(argsHandle, args, argsSize, &paramHandle));
    CHECK_ACL(aclrtKernelArgsFinalize(argsHandle));

    aclrtLaunchKernelCfg cfg{};
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = 120;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    CHECK_ACL(aclrtLaunchKernelWithConfig(func, 1, stream, &cfg, argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    return 0;
}

int TouchDoneFile(const std::string &path)
{
    std::ofstream os(path, std::ios::trunc);
    os << "done\n";
    return os.good() ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!ParseOptions(argc, argv, &opt)) {
        Usage(argv[0]);
        return 1;
    }

    const uint64_t payloadBytes = opt.message.size();
    if (payloadBytes > opt.bytes) {
        fprintf(stderr, "message length %llu exceeds --bytes capacity %llu\n",
            static_cast<unsigned long long>(payloadBytes), static_cast<unsigned long long>(opt.bytes));
        return 1;
    }

    const uint64_t mailboxBytes = sizeof(HcommSendRecvMailboxHeader) + opt.bytes;
    Log(opt, "starting: logicDev=%d phyDev=%u ip=%s bytes=%llu mailboxBytes=%llu payloadBytes=%llu",
        opt.logicDev, opt.phyDev, opt.ip.c_str(),
        static_cast<unsigned long long>(opt.bytes),
        static_cast<unsigned long long>(mailboxBytes), 
        static_cast<unsigned long long>(payloadBytes));
    Log(opt, "message: \"%s\"", opt.message.c_str());
    Log(opt, "files: local=%s peer=%s done=%s kernelJson=%s ipMap=%s",
        opt.localFile.c_str(), opt.peerFile.c_str(), opt.doneFile.c_str(), opt.kernelJson.c_str(),
        opt.ipMap.c_str());

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(opt.logicDev));

    UC::ASU::AICPUTransProvider provider(opt.kernelJson, opt.ipMap);

    // Step 1: InitEndpoint (creates local endpoint, phyDev from mapping file)
    Log(opt, "init endpoint: localIp=%s phyDev=%u", opt.ip.c_str(), opt.phyDev);
    CHECK_STATUS(provider.InitEndpoint(opt.ip));

    // Step 2: Get local EndpointDesc (same one used for endpoint creation)
    const EndpointDesc& localEp = provider.GetLocalEndpointDesc();

    // Step 3: Allocate device mailbox buffer
    void *mailboxBuf = nullptr;
    Log(opt, "allocating device mailbox buffer");
    CHECK_ACL(aclrtMalloc(&mailboxBuf, mailboxBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    CHECK_ACL(aclrtMemset(mailboxBuf, mailboxBytes, 0, mailboxBytes));
    Log(opt, "mailbox buffer ready: addr=%p size=%llu", mailboxBuf, 
        static_cast<unsigned long long>(mailboxBytes));

    std::vector<uint8_t> host(opt.message.begin(), opt.message.end());
    if (opt.rank == 0) {
        auto *payload = static_cast<uint8_t *>(mailboxBuf) + sizeof(HcommSendRecvMailboxHeader);
        Log(opt, "copying message to local mailbox payload");
        CHECK_ACL(aclrtMemcpy(payload, payloadBytes, host.data(), payloadBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        Log(opt, "receiver mailbox payload left zeroed until remote send completes");
    }

    // Step 4: Register mailbox memory (nullptr connectionHandle, uses endpoint only)
    Log(opt, "registering mailbox memory");
    std::vector<UC::ASU::TransProvider::RegisterMemoryDesc> memDescs;
    memDescs.push_back({UC::ASU::TransProvider::MemType::MEM_DEVICE, 
                       reinterpret_cast<uintptr_t>(mailboxBuf), mailboxBytes});
    std::vector<UC::ASU::TransProvider::MemHandle> memHandles;
    CHECK_STATUS(provider.RegisterMemory(nullptr, memDescs, memHandles));
    if (memHandles.empty()) {
        fprintf(stderr, "Failed to register memory\n");
        return 1;
    }
    auto memHandle = memHandles[0];
    Log(opt, "memory registered: handle=%p", memHandle);

    // Step 5: Export memory descriptor (nullptr connectionHandle)
    Log(opt, "exporting mailbox descriptor");
    void *exportDesc = nullptr;
    uint32_t exportLen = 0;
    CHECK_STATUS(provider.ExportMemory(nullptr, memHandle, &exportDesc, &exportLen));

    // Step 6: Write peer file (localEp + memory descriptor)
    CHECK_RET(WritePeerFile(opt.localFile, localEp, exportDesc, exportLen));
    Log(opt, "wrote peer file: path=%s descLen=%u", opt.localFile.c_str(), exportLen);

    // Step 7: Read peer file (get peer EndpointDesc + memory descriptor)
    EndpointDesc peerEp;
    std::vector<char> peerDesc;
    Log(opt, "waiting for peer file: path=%s", opt.peerFile.c_str());
    CHECK_RET(WaitForPeerFile(opt, opt.peerFile, &peerEp, &peerDesc));

    char peerIpBuf[64] = {};
    inet_ntop(AF_INET, &peerEp.commAddr.addr, peerIpBuf, sizeof(peerIpBuf));
    Log(opt, "loaded peer file: peerIp=%s peerPhyDev=%u descLen=%zu",
        peerIpBuf, peerEp.loc.device.devPhyId, peerDesc.size());

    // Step 8: Import peer memory (nullptr connectionHandle)
    Log(opt, "importing peer mailbox memory");
    UC::ASU::TransProvider::MemHandle peerMemHandle = nullptr;
    CHECK_STATUS(provider.ImportMemory(nullptr, peerDesc.data(), 
                                       static_cast<uint32_t>(peerDesc.size()), &peerMemHandle));
    Log(opt, "peer memory imported: handle=%p", peerMemHandle);

    uint64_t peerMemAddr = 0;
    uint64_t peerMemSize = 0;
    CHECK_STATUS(provider.GetImportedMemoryInfo(nullptr, peerMemHandle, &peerMemAddr, &peerMemSize));
    Log(opt, "mailbox mapping: localMailbox=%p localSize=%llu peerMailbox=0x%llx peerSize=%llu",
        mailboxBuf, static_cast<unsigned long long>(mailboxBytes),
        static_cast<unsigned long long>(peerMemAddr),
        static_cast<unsigned long long>(peerMemSize));

    // Step 9: Create connection (remoteIp resolved via IP-to-device mapping)
    Log(opt, "creating connection: local=%s peer=%s port=%u", opt.ip.c_str(), peerIpBuf, kDefaultPort);
    std::vector<UC::ASU::TransProvider::ConnectionHandle> connections;
    CHECK_STATUS(provider.CreateConnection(opt.ip, peerIpBuf, kDefaultPort, 1, 5000, connections));
    if (connections.empty()) {
        fprintf(stderr, "Failed to create connection\n");
        return 1;
    }
    auto connection = connections[0];
    Log(opt, "connection created: handle=%p channel=%llu", connection,
        static_cast<unsigned long long>(provider.GetChannelHandle(connection)));

    // Step 10: Allocate AICPU thread
    Log(opt, "allocating AICPU TS thread");
    std::vector<uint32_t> notifyNumPerThread = {0};
    std::vector<UC::ASU::TransProvider::ThreadHandle> threads;
    CHECK_STATUS(provider.AllocThread(1, notifyNumPerThread, threads));
    if (threads.empty()) {
        fprintf(stderr, "Failed to allocate thread\n");
        return 1;
    }
    auto thread = threads[0];
    Log(opt, "thread allocated: handle=%p", thread);

    // Step 11: Build sendRecvContext
    HcommSendRecvChannelContext sendRecvContext{};
    sendRecvContext.magic = HCOMM_SEND_RECV_CHANNEL_CONTEXT_MAGIC;
    sendRecvContext.version = HCOMM_SEND_RECV_CHANNEL_CONTEXT_VERSION;
    sendRecvContext.transportChannel = provider.GetChannelHandle(connection);
    sendRecvContext.localMailboxAddr = reinterpret_cast<uint64_t>(mailboxBuf);
    sendRecvContext.localMailboxSize = mailboxBytes;
    sendRecvContext.remoteMailboxAddr = peerMemAddr;
    sendRecvContext.remoteMailboxSize = peerMemSize;

    void *sendRecvContextBuf = nullptr;
    CHECK_ACL(aclrtMalloc(&sendRecvContextBuf, sizeof(sendRecvContext), ACL_MEM_MALLOC_HUGE_ONLY));
    CHECK_ACL(aclrtMemcpy(sendRecvContextBuf, sizeof(sendRecvContext), &sendRecvContext, 
                          sizeof(sendRecvContext), ACL_MEMCPY_HOST_TO_DEVICE));
    Log(opt, "sendRecvContext ready: addr=%p transportChannel=%llu",
        sendRecvContextBuf, static_cast<unsigned long long>(sendRecvContext.transportChannel));

    // Step 12: Launch kernel
    if (opt.rank == 0) {
        aclrtStream stream = nullptr;
        CHECK_ACL(aclrtCreateStream(&stream));
        aclrtBinHandle bin = nullptr;
        aclrtFuncHandle func = nullptr;
        CHECK_RET(LoadKernel(opt.kernelJson, "HixlSend", &bin, &func));

        HixlSendParam args{};
        args.thread = reinterpret_cast<uint64_t>(thread);
        args.channel = reinterpret_cast<uint64_t>(sendRecvContextBuf);
        args.local_src = static_cast<uint8_t *>(mailboxBuf) + sizeof(HcommSendRecvMailboxHeader);
        args.len = payloadBytes;
        Log(opt, "launching HixlSend: thread=%llu channel=%p src=%p len=%llu",
            static_cast<unsigned long long>(args.thread), sendRecvContextBuf, args.local_src,
            static_cast<unsigned long long>(args.len));
        CHECK_RET(LaunchKernel(func, stream, &args, sizeof(args)));

        PrintBytes(opt, "expected payload", host.data(), payloadBytes);
        Log(opt, "HixlSend completed, writing done file");
        CHECK_RET(TouchDoneFile(opt.doneFile));
        CHECK_ACL(aclrtDestroyStream(stream));
    } else {
        aclrtStream stream = nullptr;
        CHECK_ACL(aclrtCreateStream(&stream));
        aclrtBinHandle bin = nullptr;
        aclrtFuncHandle func = nullptr;
        CHECK_RET(LoadKernel(opt.kernelJson, "HixlRecv", &bin, &func));

        void *recvBuf = nullptr;
        void *receivedLenBuf = nullptr;
        CHECK_ACL(aclrtMalloc(&recvBuf, opt.bytes, ACL_MEM_MALLOC_HUGE_ONLY));
        CHECK_ACL(aclrtMemset(recvBuf, opt.bytes, 0, opt.bytes));
        CHECK_ACL(aclrtMalloc(&receivedLenBuf, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_ONLY));
        CHECK_ACL(aclrtMemset(receivedLenBuf, sizeof(uint64_t), 0, sizeof(uint64_t)));

        HixlRecvParam args{};
        args.thread = reinterpret_cast<uint64_t>(thread);
        args.channel = reinterpret_cast<uint64_t>(sendRecvContextBuf);
        args.local_dst = recvBuf;
        args.dst_capacity = opt.bytes;
        args.received_len = static_cast<uint64_t *>(receivedLenBuf);
        args.timeout_ms = 120000;
        Log(opt, "launching HixlRecv: thread=%llu channel=%p dst=%p capacity=%llu timeoutMs=%u",
            static_cast<unsigned long long>(args.thread), sendRecvContextBuf, args.local_dst,
            static_cast<unsigned long long>(args.dst_capacity), args.timeout_ms);
        const int recvLaunchRet = LaunchKernel(func, stream, &args, sizeof(args));
        Log(opt, "HixlRecv returned ret=%d", recvLaunchRet);

        if (recvLaunchRet != 0) {
            return Fail("LaunchKernel(HixlRecv)", recvLaunchRet, __LINE__);
        }

        uint64_t receivedLen = 0;
        CHECK_ACL(aclrtMemcpy(&receivedLen, sizeof(receivedLen), receivedLenBuf, 
                              sizeof(receivedLen), ACL_MEMCPY_DEVICE_TO_HOST));
        if (receivedLen > opt.bytes) {
            fprintf(stderr, "verify failed: receivedLen %llu exceeds capacity %llu\n",
                static_cast<unsigned long long>(receivedLen), static_cast<unsigned long long>(opt.bytes));
            return 1;
        }
        if (receivedLen != payloadBytes) {
            fprintf(stderr, "verify failed: expected receivedLen %llu, got %llu\n",
                static_cast<unsigned long long>(payloadBytes), 
                static_cast<unsigned long long>(receivedLen));
            return 1;
        }

        Log(opt, "HixlRecv completed, copying received payload back to host");
        std::vector<uint8_t> got(receivedLen, 0);
        CHECK_ACL(aclrtMemcpy(got.data(), receivedLen, recvBuf, receivedLen, ACL_MEMCPY_DEVICE_TO_HOST));
        const std::string received(got.begin(), got.end());
        if (got != host) {
            fprintf(stderr, "verify failed: expected \"%s\", got \"%s\"\n", 
                    opt.message.c_str(), received.c_str());
            return 1;
        }
        Log(opt, "verify ok: received \"%s\" (%llu bytes)",
            received.c_str(), static_cast<unsigned long long>(receivedLen));
        CHECK_ACL(aclrtFree(receivedLenBuf));
        CHECK_ACL(aclrtFree(recvBuf));
        CHECK_ACL(aclrtDestroyStream(stream));
    }

    // Cleanup
    Log(opt, "releasing resources");
    CHECK_VEC_STATUS(provider.FreeThread(threads));
    CHECK_STATUS(provider.UnimportMemory(nullptr, peerMemHandle));
    {
        std::vector<UC::ASU::TransProvider::UnregisterMemoryDesc> unregDescs;
        for (auto& mh : memHandles) {
            unregDescs.push_back({nullptr, mh});
        }
        CHECK_VEC_STATUS(provider.UnregisterMemory(unregDescs));
    }
    CHECK_VEC_STATUS(provider.DeleteConnections(connections));
    CHECK_ACL(aclrtFree(sendRecvContextBuf));
    CHECK_ACL(aclrtFree(mailboxBuf));
    CHECK_ACL(aclrtResetDevice(opt.logicDev));
    CHECK_ACL(aclFinalize());
    Log(opt, "done");
    return 0;
}