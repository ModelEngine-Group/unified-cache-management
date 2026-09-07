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

#include <arpa/inet.h>
#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <sys/types.h>

namespace umc::comm::v2 {

constexpr int32_t DEV_EID_INFO_MAX_NAME = 64;
constexpr int32_t DEV_QP_KEY_SIZE = 64;
constexpr int32_t HCCP_MAX_TPID_INFO_NUM = 128;
constexpr int32_t CUSTOM_CHAN_DATA_MAX_SIZE = 2048;
constexpr int32_t MAX_INTERFACE_NUM = 8;
constexpr uint32_t TOKEN_VALUE = 0;
constexpr int32_t MEM_KEY_SIZE = 128;
constexpr int32_t URMA_TOKEN_PLAIN_TEXT = 1;
constexpr uint32_t HCCP_SOCK_CONN_TAG_SIZE = 192;
constexpr uint32_t HCCP_MAX_INTERFACE_NAME_LEN = 256;

constexpr int32_t HCCP_OTHERS_EAGAIN = 128301;
constexpr int32_t HCCP_SOCK_EAGAIN = 128201;

enum HccpNetworkMode {
    NETWORK_PEER_ONLINE = 0,
    NETWORK_OFFLINE = 1,
    NETWORK_ONLINE = 2,
};

enum HccpNotifyType {
    NO_USE = 0,
    NOTIFY = 1,
    EVENTID = 2,
};

union HccpIpAddr {
    struct in_addr addr;
    struct in6_addr addr6;
};

enum DrvHdcServiceType : int {
    HDC_SERVICE_TYPE_RDMA = 6,
    HDC_SERVICE_TYPE_RDMA_V2 = 18,
};

struct RaInitConfig {
    unsigned int phyId;
    HccpNetworkMode nicPosition;
    DrvHdcServiceType hdcType;
    bool enableHdcAsync;
};

struct MemKey {
    uint8_t value[MEM_KEY_SIZE];
    uint8_t size;
};

enum SubProcType {
    TSD_SUB_PROC_HCCP = 0,
    TSD_SUB_PROC_COMPUTE = 1,
    TSD_SUB_PROC_CUSTOM_COMPUTE = 2,
    TSD_SUB_PROC_QUEUE_SCHEDULE = 3,
    TSD_SUB_PROC_UDF = 4,
    TSD_SUB_PROC_NPU = 5,
    TSD_SUB_PROC_PROXY = 6,
    TSD_SUB_PROC_BUILTIN_UDF = 7,
    TSD_SUB_PROC_ADPROF = 8,
    TSD_SUB_PROC_MAX = 0xFF,
};

struct ProcEnvParam {
    const char* envName;
    uint64_t nameLen;
    const char* envValue;
    uint64_t valueLen;
};

struct ProcExtParam {
    const char* paramInfo;
    uint64_t paramLen;
};

struct ProcOpenArgs {
    SubProcType procType;
    ProcEnvParam* envParaList;
    uint64_t envCnt;
    const char* filePath;
    uint64_t pathLen;
    ProcExtParam* extParamList;
    uint64_t extParamCnt;
    pid_t* subPid;
};

union HccpEid {
    uint8_t raw[16];
    struct {
        uint64_t reserved;
        uint32_t prefix;  // == 0
        uint32_t addr;    // == IPv4 address
    } in4;
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } in6;
};
static_assert(sizeof(HccpEid) == 16, "HccpEid must be 16 bytes");

struct RaInfo {
    HccpNetworkMode mode;
    unsigned int phyId;
};

struct DevEidInfo {
    char name[DEV_EID_INFO_MAX_NAME];
    uint32_t type;
    uint32_t eidIndex;
    HccpEid eid;
    uint32_t dieId;
    uint32_t chipId;
    uint32_t funcId;
    uint32_t resv;
};

struct CtxInitCfg {
    HccpNetworkMode mode;
};

struct CtxInitAttr {
    unsigned int phyId;
    union {
        struct {
            HccpNotifyType notifyType;
            int family;
            HccpIpAddr localIp;
        } rdma;
        struct {
            uint32_t eidIndex;
            uint32_t resv0;
            HccpEid eid;
        } ub;
    };
    uint32_t resv[16];
};

struct DevNotifyInfo {
    uint64_t va;
    uint64_t size;
    MemKey key;
    uint32_t resv[4];
};

struct DevBaseAttrT {
    uint32_t sqMaxDepth;
    uint32_t rqMaxDepth;
    uint32_t sqMaxSge;
    uint32_t rqMaxSge;
    union {
        struct {
            DevNotifyInfo globalNotifyInfo;
        } rdma;
        struct {
            uint32_t maxJfsInlineLen;
            uint32_t maxJfsRsge;
            uint32_t dieId;
            uint32_t chipId;
            uint32_t funcId;
        } ub;
    } devInfo;
    uint32_t resv[16];
};

union DataPlaneCstmFlag {
    struct {
        uint32_t poolCqCstm : 1;  // 0: hccp poll cq; 1: caller poll cq
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

struct ChanInfoT {
    struct {
        DataPlaneCstmFlag dataPlaneFlag;
    } in;
    struct {
        int fd;
    } out;
};

enum JfcMode {
    JFC_MODE_NORMAL = 0,
    JFC_MODE_STARS_POLL = 1,
    JFC_MODE_CCU_POLL = 2,
    JFC_MODE_USER_CTL_NORMAL = 3,
    JFC_MODE_MAX,
};

union JfcFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t jfcInline : 1;
        uint32_t reserved : 30;
    } bs;
    uint32_t value;
};

struct CqCreateAttr {
    void* chanHandle;
    uint32_t depth;
    union {
        struct {
            uint64_t cqContext;
            uint32_t mode;
            uint32_t compVector;
        } rdma;
        struct {
            uint64_t userCtx;
            JfcMode mode;
            uint32_t ceqn;
            JfcFlag flag;
            struct {
                bool valid;
                uint32_t cqeFlag;
            } ccuExCfg;
        } ub;
    };
};

struct CqCreateInfo {
    uint64_t va;
    uint32_t id;
    uint32_t cqeSize;
    uint64_t bufAddr;
    uint64_t swdbAddr;
};

struct CqInfoT {
    CqCreateAttr in;
    CqCreateInfo out;
};

enum JettyMode {
    JETTY_MODE_URMA_NORMAL = 0,
    JETTY_MODE_CACHE_LOCK_DWQE = 1,
    JETTY_MODE_CCU = 2,
    JETTY_MODE_USER_CTL_NORMAL = 3,
    JETTY_MODE_CCU_TA_CACHE = 4,
    JETTY_MODE_MAX,
};

enum TransportModeT {
    CONN_RM = 1,
    CONN_RC = 2,
};

union JettyFlag {
    struct {
        uint32_t shareJfr : 1;
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

union JfsFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t errorSuspend : 1;
        uint32_t outorderComp : 1;
        uint32_t orderType : 8;
        uint32_t multiPath : 1;
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

struct JettyQueCfgEx {
    uint32_t buffSize;
    uint64_t buffVa;
};

union CstmJfsFlag {
    struct {
        uint32_t sqCstm : 1;
        uint32_t dbCstm : 1;
        uint32_t dbCtlCstm : 1;
        uint32_t reserved : 29;
    } bs;
    uint32_t value;
};

struct QpCreateAttr {
    void* scqHandle;
    void* rcqHandle;
    void* srqHandle;
    uint32_t sqDepth;
    uint32_t rqDepth;
    TransportModeT transportMode;
    union {
        struct {
            uint32_t mode;
            uint32_t udpSport;
            uint8_t trafficClass;
            uint8_t sl;
            uint8_t timeout;
            uint8_t rnrRetry;
            uint8_t retryCnt;
        } rdm;
        struct {
            JettyMode mode;
            uint32_t jettyId;
            JettyFlag flag;
            JfsFlag jfsFlag;
            void* tokenIdHandle;
            uint32_t tokenValue;
            uint8_t priority;
            uint8_t rnrRetry;
            uint8_t errTimeout;
            union {
                struct {
                    JettyQueCfgEx sq;
                    bool piType;
                    CstmJfsFlag cstmFlag;
                    uint32_t sqebbNum;
                } extMode;
                struct {
                    bool lockFlag;
                    uint32_t sqeBufIdx;
                } taCacheMode;
            };
        } ub;
    };
    uint32_t resv[16];
};

struct QpKeyT {
    uint8_t value[DEV_QP_KEY_SIZE];
    uint8_t size;
};

struct QpCreateInfo {
    QpKeyT key;
    union {
        struct {
            uint32_t qpn;
        } rdma;
        struct {
            uint32_t uasid;
            uint32_t id;
            uint64_t sqBuffVa;
            uint64_t wqebbSize;
            uint64_t dbAddr;
            uint32_t dbTokenId;
            uint64_t ciAddr;
        } ub;
    };
    uint64_t va;
    uint32_t resv[16];
};

struct HccpTokenId {
    uint32_t tokenId;
};

enum TokenPolicyV2 : uint32_t {
    TOKEN_POLICY_NONE = 0,
    TOKEN_POLICY_PLAIN_TEXT = 1,
    TOKEN_POLICY_SIGNED = 2,
    TOKEN_POLICY_ALL_ENCRYPTED = 3,
    TOKEN_POLICY_RESERVED,
};

union ImportJettyFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t orderType : 8;
        uint32_t shareTp : 1;
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

enum JettyGrpPolicy : uint32_t {
    JETTY_GRP_POLICY_RR = 0,
    JETTY_GRP_POLICY_HASH_HINT = 1,
    JETTY_GRP_POLICY_MAX,
};

enum TargetType {
    TARGET_TYPE_JFR = 0,
    TARGET_TYPE_JETTY = 1,
    TARGET_TYPE_JETTY_GROUP = 2,
    TARGET_TYPE_MAX,
};

enum JettyImportMode {
    JETTY_IMPORT_MODE_NORMAL = 0,
    JETTY_IMPORT_MODE_EXP = 1,
    JETTY_IMPORT_MODE_MAX,
};

struct JettyImportExpCfg {
    uint64_t tpHandle;
    uint64_t peerTpHandle;
    uint64_t tag;
    uint32_t txPsn;
    uint32_t rxPsn;
    uint32_t rsv[16];
};

struct QpImportAttr {
    QpKeyT key;
    union {
        struct {
            JettyImportMode mode;
            uint32_t tokenValue;
            JettyGrpPolicy policy;
            TargetType type;
            ImportJettyFlag flag;
            JettyImportExpCfg expImportCfg;
            uint32_t tpType;
        } ub;
    };
    uint32_t resv[7];
};

struct QpImportInfo {
    union {
        struct {
            uint64_t tjettyHandle;
            uint32_t tpn;
        } ub;
    };
    uint32_t resv[8];
};

struct QpImportInfoT {
    QpImportAttr in;
    QpImportInfo out;
};

enum MemSegTokenPolicy {
    MEM_SEG_TOKEN_NONE = 0,
    MEM_SEG_TOKEN_PLAIN_TEXT = 1,
    MEM_SEG_TOKEN_SIGNED = 2,
    MEM_SEG_TOKEN_ALL_ENCRYPTED = 3,
    MEM_SEG_TOKEN_RESERVED = 4,
};

enum MemSegAccessFlags {
    MEM_SEG_ACCESS_LOCAL_ONLY = 1,
    MEM_SEG_ACCESS_READ = (1 << 1),
    MEM_SEG_ACCESS_WRITE = (1 << 2),
    MEM_SEG_ACCESS_ATOMIC = (1 << 3),
    MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC,
};

struct MemInfo {
    uint64_t addr;
    uint64_t size;
};

union RegSegFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t cacheable : 1;
        uint32_t dsva : 1;
        uint32_t access : 6;
        uint32_t nonPin : 1;
        uint32_t userIova : 1;
        uint32_t tokenIdValid : 1;
        uint32_t reserved : 18;
    } bs;
    uint32_t value;
};

struct MemRegAttr {
    MemInfo mem;
    union {
        struct {
            int access;
        } rdma;
        struct {
            RegSegFlag flags;
            uint32_t tokenValue;
            void* tokenIdHandle;
        } ub;
    };
    uint32_t resv[8];
};

struct MemRegInfo {
    MemKey key;
    union {
        struct {
            uint32_t lkey;
        } rdma;
        struct {
            uint32_t tokenId;
            uint64_t targetSegHandle;
        } ub;
    };
    uint32_t resv[8];
};

struct MrRegInfoT {
    MemRegAttr in;
    MemRegInfo out;
};

union ImportSegFlag {
    struct {
        uint32_t cacheable : 1;
        uint32_t access : 6;
        uint32_t mapping : 1;
        uint32_t reserved : 24;
    } bs;
    uint32_t value;
};

struct MemImportAttr {
    MemKey key;
    union {
        struct {
            ImportSegFlag flags;
            uint64_t mappingAddr;
            uint32_t tokenValue;
        } ub;
    };
    uint32_t resv[4];
};

struct MemImportInfo {
    union {
        struct {
            uint32_t key;
        } rdma;
        struct {
            uint64_t targetSegHandle;
        } ub;
    };
    uint32_t resv[4];
};

struct MrImportInfoT {
    MemImportAttr in;
    MemImportInfo out;
};

enum JettyAttrMask : uint32_t {
    JETTY_ATTR_RX_THRESHOLD = 0x01,
    JETTY_ATTR_STATE = (0x01 << 1),
};

enum JettyState {
    JETTY_STATE_RESET = 0,
    JETTY_STATE_READY = 1,
    JETTY_STATE_SUSPENDED = 2,
    JETTY_STATE_ERROR = 3,
};

struct JettyAttr {
    JettyAttrMask mask;
    uint32_t rxThreshold;
    JettyState state;
    uint32_t resv[2];
};

struct WrSgeList {
    uint64_t addr;
    uint32_t len;
    void* lmemHandle;
};

enum RaWrOpcode {
    RA_WR_RDMA_WRITE = 0,
    RA_WR_RDMA_WRITE_WITH_IMM = 1,
    RA_WR_SEND = 2,
    RA_WR_SEND_WITH_IMM = 3,
    RA_WR_RDMA_READ = 4,
    RA_WR_RDMA_ATOMIC_WRITE = 0xf0,
    RA_WR_RDMA_WRITE_WITH_NOTIFY = 0xf2,
    RA_WR_RDMA_REDUCE_WRITE = 0xf5,
    RA_WR_RDMA_REDUCE_WRITE_WITH_NOTIFY = 0xf6,
};

struct WrAuxInfo {
    uint8_t dataType;
    uint8_t reduceType;
    uint32_t notifyOffset;
};

struct WrNotifyInfo {
    uint64_t notifyData;
    uint64_t notifyAddr;
    void* notifyHandle;
};

struct WrReduceInfo {
    bool reduceEn;
    uint8_t reduceOpcode;
    uint8_t reduceDataType;
};

enum RaUbOpcode {
    RA_UB_OPC_WRITE = 0x00,
    RA_UB_OPC_WRITE_NOTIFY = 0x02,
    RA_UB_OPC_READ = 0x10,
    RA_UB_OPC_NOP = 0x51,
    RA_UB_OPC_LAST = 0x00,
};

union JfsWrFlag {
    struct {
        uint32_t placeOrder : 2;
        uint32_t compOrder : 1;
        uint32_t fence : 1;
        uint32_t solicitedEnable : 1;
        uint32_t completeEnable : 1;
        uint32_t inlineFlag : 1;
        uint32_t reserved : 25;
    } bs;
    uint32_t value;
};

struct SendWrData {
    WrSgeList* sges;
    uint32_t numSge;
    uint8_t* inlineData;
    uint32_t inlineSize;
    uint64_t remoteAddr;
    void* rmemHandle;
    union {
        struct {
            uint64_t wrId;
            RaWrOpcode opcode;
            unsigned int flags;
            WrAuxInfo aux;
        } rdma;
        struct {
            uint64_t userCtx;
            RaUbOpcode opcode;
            JfsWrFlag flags;
            void* rem_qp_handle;
            WrNotifyInfo notifyInfo;
            WrReduceInfo reduceInfo;
        } ub;
    };
    uint32_t immData;
    uint32_t resv[10];
};

struct UbPostInfo {
    uint16_t funcId : 7;
    uint16_t dieId : 1;
    uint16_t rsv : 8;
    uint16_t jettyId;
    uint16_t piVal;
    uint8_t dwqe[128];
    uint16_t dwqeSize;
};

struct WqeInfoT {
    unsigned int sqIndex;
    unsigned int wqeIndex;
};

struct DbInfo {
    unsigned int dbIndex;
    unsigned long dbInfo;
};

struct SendWrResp {
    union {
        WqeInfoT wqeTmp;
        DbInfo db;
        UbPostInfo doorbellInfo;
        uint8_t resv[384];
    };
};

struct CustomChanInfoIn {
    char data[CUSTOM_CHAN_DATA_MAX_SIZE];
    unsigned int offsetStart;
    unsigned int op;
};

struct CustomChanInfoOut {
    char data[CUSTOM_CHAN_DATA_MAX_SIZE];
    unsigned int offsetNext;
    int opRet;
};

union GetTpCfgFlag {
    struct {
        uint32_t ctp : 1;
        uint32_t rtp : 1;
        uint32_t utp : 1;
        uint32_t uboe : 1;
        uint32_t preDefined : 1;
        uint32_t dynamicDefined : 1;
        uint32_t reserved : 26;
    } bs;
    uint32_t value;
};

struct GetTpCfg {
    GetTpCfgFlag flag;
    TransportModeT transMode;
    HccpEid localEid;
    HccpEid peerEid;
};

struct TpInfo {
    uint64_t tpHandle;
    uint32_t resv;
};

struct RaGetIfAttr {
    unsigned int phyId;
    HccpNetworkMode nicPosition;
    bool isAll;
};

struct IfAddrInfo {
    HccpIpAddr ip;
    struct in_addr mask;
};

struct InterfaceInfo {
    int family;
    int scopeId;
    IfAddrInfo ifAddr;
    char ifName[HCCP_MAX_INTERFACE_NAME_LEN];
};
static_assert(sizeof(InterfaceInfo) == 284, "InterfaceInfo HCCP ABI size mismatch");
static_assert(offsetof(InterfaceInfo, family) == 0, "InterfaceInfo.family ABI mismatch");
static_assert(offsetof(InterfaceInfo, scopeId) == 4, "InterfaceInfo.scopeId ABI mismatch");
static_assert(offsetof(InterfaceInfo, ifAddr) == 8, "InterfaceInfo.ifAddr ABI mismatch");
static_assert(offsetof(InterfaceInfo, ifName) == 28, "InterfaceInfo.ifName ABI mismatch");

constexpr int SOCK_EAGAIN = 128201;
constexpr int SOCK_CLOSE = 128203;
constexpr int SOCK_ESOCKCLOSED = 128207;
constexpr int HCCP_SOCKET_CONNECTED = 1;  // SocketInfoT.status: 1=connected

enum IdType {
    PHY_ID_VNIC_IP = 0,
    SDID_VNIC_IP = 1,
};

struct IpInfo {
    int family;
    HccpIpAddr ip;
    uint32_t resv[2];
};
static_assert(sizeof(IpInfo) == 28, "IpInfo HCCP ABI size mismatch");
static_assert(offsetof(IpInfo, family) == 0, "IpInfo.family ABI mismatch");
static_assert(offsetof(IpInfo, ip) == 4, "IpInfo.ip ABI mismatch");
static_assert(offsetof(IpInfo, resv) == 20, "IpInfo.resv ABI mismatch");

struct Rdev {
    unsigned int phyId;
    int family;  // AF_INET / AF_INET6
    HccpIpAddr localIp;
};

struct SocketConnectInfoT {
    void* socketHandle;
    HccpIpAddr remoteIp;
    unsigned int port;
    char tag[HCCP_SOCK_CONN_TAG_SIZE];
};

struct SocketCloseInfoT {
    void* socketHandle;
    void* fdHandle;
    int disuseLinger;
};

struct SocketInfoT {
    void* socketHandle;
    void* fdHandle;
    HccpIpAddr remoteIp;
    int status;  // 0 not connected / 1 connected / 2 timeout / 3 connecting
    char tag[HCCP_SOCK_CONN_TAG_SIZE];
};

}  // namespace umc::comm::v2
