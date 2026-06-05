/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_ACL_ACL_RT_H_
#define INC_EXTERNAL_ACL_ACL_RT_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "acl_base.h"
#include "acl_rt_allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

// Current version is 1.17.0
#define ACL_MAJOR_VERSION              1
#define ACL_MINOR_VERSION              17
#define ACL_PATCH_VERSION              0
#define ACL_EVENT_SYNC                    0x00000001U
#define ACL_EVENT_CAPTURE_STREAM_PROGRESS 0x00000002U
#define ACL_EVENT_TIME_LINE               0x00000008U
#define ACL_EVENT_DEVICE_USE_ONLY         0x00000010U
#define ACL_EVENT_EXTERNAL                0x00000020U
#define ACL_EVENT_IPC                     0x00000040U

// for create stream
#define ACL_STREAM_FAST_LAUNCH      0x00000001U
#define ACL_STREAM_FAST_SYNC        0x00000002U
#define ACL_STREAM_PERSISTENT       0x00000004U
#define ACL_STREAM_HUGE             0x00000008U
#define ACL_STREAM_CPU_SCHEDULE     0x00000010U
#define ACL_STREAM_DEVICE_USE_ONLY  0x00000020U

#define ACL_STREAM_WAIT_VALUE_GEQ 0x00000000U
#define ACL_STREAM_WAIT_VALUE_EQ  0x00000001U
#define ACL_STREAM_WAIT_VALUE_AND 0x00000002U
#define ACL_STREAM_WAIT_VALUE_NOR 0x00000003U

#define ACL_CONTINUE_ON_FAILURE 0x00000000U
#define ACL_STOP_ON_FAILURE     0x00000001U

// for notify | for create notify
#define ACL_NOTIFY_DEFAULT          0x00000000U
#define ACL_NOTIFY_DEVICE_USE_ONLY  0x00000001U

// for device get capability
#define ACL_DEV_FEATURE_SUPPORT     0x00000001
#define ACL_DEV_FEATURE_NOT_SUPPORT 0x00000000

#define ACL_RT_NOTIFY_EXPORT_FLAG_DEFAULT                0x0UL
#define ACL_RT_NOTIFY_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x02UL

#define ACL_RT_NOTIFY_IMPORT_FLAG_DEFAULT            0x0UL
#define ACL_RT_NOTIFY_IMPORT_FLAG_ENABLE_PEER_ACCESS 0x02UL

#define ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT                0x0UL
#define ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x1UL

#define ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT            0x0UL
#define ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS 0x1UL

#define ACL_RT_VMM_EXPORT_FLAG_DEFAULT                0x0UL
#define ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION 0x1UL

// Host register flags for aclrtHostRegisterV2
#define ACL_HOST_REG_MAPPED   0x2UL    // Map host memory to device address
#define ACL_HOST_REG_IOMEMORY 0x4UL    // Register as IO memory
#define ACL_HOST_REG_READONLY 0x8UL    // Register as read-only memory
#define ACL_HOST_REG_PINNED   0x10000000UL  // Pin memory to prevent swapping

// flags for aclmdlRIDebugJsonPrint
#define ACL_MDLRI_DEBUG_JSON_PRINT_SUMMARY 0x0UL
#define ACL_MDLRI_DEBUG_JSON_PRINT_VERBOSE 0x1UL

#define ACL_RT_MAX_RECORD_PA_NUM_PER_DEV 20U

#define ACL_IPC_EVENT_HANDLE_SIZE 64U

// for uvm memory
#define ACL_RT_MEM_ATTACH_GLOBAL 0x01U

typedef enum aclrtRunMode {
    ACL_DEVICE,
    ACL_HOST,
} aclrtRunMode;

typedef enum aclrtTsId {
    ACL_TS_ID_AICORE   = 0,
    ACL_TS_ID_AIVECTOR = 1,
    ACL_TS_ID_RESERVED = 2,
} aclrtTsId;

typedef enum aclrtEventStatus {
    ACL_EVENT_STATUS_COMPLETE  = 0,
    ACL_EVENT_STATUS_NOT_READY = 1,
    ACL_EVENT_STATUS_RESERVED  = 2,
} aclrtEventStatus;

typedef enum aclrtEventRecordedStatus {
    ACL_EVENT_RECORDED_STATUS_NOT_READY = 0,
    ACL_EVENT_RECORDED_STATUS_COMPLETE = 1,
} aclrtEventRecordedStatus;

typedef enum aclrtEventWaitStatus {
    ACL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    ACL_EVENT_WAIT_STATUS_NOT_READY = 1,
    ACL_EVENT_WAIT_STATUS_RESERVED  = 0xFFFF,
} aclrtEventWaitStatus;

typedef enum aclrtStreamStatus {
    ACL_STREAM_STATUS_COMPLETE  = 0,
    ACL_STREAM_STATUS_NOT_READY = 1,
    ACL_STREAM_STATUS_RESERVED  = 0xFFFF,
} aclrtStreamStatus;

typedef enum aclrtCallbackBlockType {
    ACL_CALLBACK_NO_BLOCK,
    ACL_CALLBACK_BLOCK,
} aclrtCallbackBlockType;

typedef enum aclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEFAULT,
    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE,
    ACL_MEMCPY_INNER_DEVICE_TO_DEVICE,
    ACL_MEMCPY_INTER_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemManagedAdviseType {
    ACL_MEM_ADVISE_SET_READ_MOSTLY = 0,
    ACL_MEM_ADVISE_UNSET_READ_MOSTLY,
    ACL_MEM_ADVISE_SET_PREFERRED_LOCATION,
    ACL_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
    ACL_MEM_ADVISE_SET_ACCESSED_BY,
    ACL_MEM_ADVISE_UNSET_ACCESSED_BY,
} aclrtMemManagedAdviseType;

typedef enum aclrtMemManagedLocationType {
    ACL_MEM_LOCATIONTYPE_INVALID = 0,
    ACL_MEM_LOCATIONTYPE_DEVICE,
    ACL_MEM_LOCATIONTYPE_HOST,
    ACL_MEM_LOCATIONTYPE_HOST_NUMA,
    ACL_MEM_LOCATIONTYPE_HOST_NUMA_CURRENT,
} aclrtMemManagedLocationType;

typedef struct aclrtMemManagedLocation{
    aclrtMemManagedLocationType type;
    int32_t id;
} aclrtMemManagedLocation;

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
    ACL_MEM_MALLOC_HUGE_FIRST_P2P,
    ACL_MEM_MALLOC_HUGE_ONLY_P2P,
    ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
    ACL_MEM_MALLOC_HUGE1G_ONLY,
    ACL_MEM_MALLOC_HUGE1G_ONLY_P2P,
    ACL_MEM_TYPE_LOW_BAND_WIDTH   = 0x0100,
    ACL_MEM_TYPE_HIGH_BAND_WIDTH  = 0x1000,
    ACL_MEM_ACCESS_USER_SPACE_READONLY = 0x100000,
} aclrtMemMallocPolicy;

typedef enum {
    ACL_HOST_REGISTER_MAPPED = 0,
    ACL_HOST_REGISTER_IOMEMORY = 0x04,
    ACL_HOST_REGISTER_READONLY = 0x08
} aclrtHostRegisterType;

typedef enum {
    ACL_RT_MEM_ATTR_RSV = 0,
    ACL_RT_MEM_ATTR_MODULE_ID,
    ACL_RT_MEM_ATTR_DEVICE_ID,
    ACL_RT_MEM_ATTR_VA_FLAG,
} aclrtMallocAttrType;

typedef union {
    uint16_t moduleId;
    uint32_t deviceId;
    uint32_t vaFlag;
    uint8_t rsv[8];
} aclrtMallocAttrValue;

typedef struct {
    aclrtMallocAttrType attr;
    aclrtMallocAttrValue value;
} aclrtMallocAttribute;

typedef struct {
    aclrtMallocAttribute* attrs;
    size_t numAttrs;
} aclrtMallocConfig;

typedef struct {
    uint32_t sdid;  // whitelisted 
    int32_t *pid;
    size_t num;
} aclrtServerPid;

typedef enum aclrtMemAttr {
    ACL_DDR_MEM,
    ACL_HBM_MEM,
    ACL_DDR_MEM_HUGE,
    ACL_DDR_MEM_NORMAL,
    ACL_HBM_MEM_HUGE,
    ACL_HBM_MEM_NORMAL,
    ACL_DDR_MEM_P2P_HUGE,
    ACL_DDR_MEM_P2P_NORMAL,
    ACL_HBM_MEM_P2P_HUGE,
    ACL_HBM_MEM_P2P_NORMAL,
    ACL_HBM_MEM_HUGE1G,
    ACL_HBM_MEM_P2P_HUGE1G,
    ACL_MEM_NORMAL,
    ACL_MEM_HUGE,
    ACL_MEM_HUGE1G,
    ACL_MEM_P2P_NORMAL,
    ACL_MEM_P2P_HUGE,
    ACL_MEM_P2P_HUGE1G,
} aclrtMemAttr;

enum aclrtMemPgType {
    NORMAL_PAGE_TYPE = 0U,
    HUGE_PAGE_TYPE,
    HUGE1G_PAGE_TYPE,
};

enum aclrtMemType {
    HBM_TYPE = 0U,
    DDR_TYPE,
    P2P_HBM_TYPE,
    P2P_DDR_TYPE,
};

// for ACL_RT_IPC_MEM_ATTR_ACCESS_LINK value
#define ACL_RT_IPC_MEM_ATTR_ACCESS_LINK_SIO 0
#define ACL_RT_IPC_MEM_ATTR_ACCESS_LINK_HCCS 1

typedef enum {
    ACL_RT_IPC_MEM_ATTR_ACCESS_LINK,
} aclrtIpcMemAttrType;

typedef enum aclrtGroupAttr {
    ACL_GROUP_AICORE_INT,
    ACL_GROUP_AIV_INT,
    ACL_GROUP_AIC_INT,
    ACL_GROUP_SDMANUM_INT,
    ACL_GROUP_ASQNUM_INT,
    ACL_GROUP_GROUPID_INT
} aclrtGroupAttr;

typedef enum aclrtFloatOverflowMode {
    ACL_RT_OVERFLOW_MODE_SATURATION = 0,
    ACL_RT_OVERFLOW_MODE_INFNAN,
    ACL_RT_OVERFLOW_MODE_UNDEF,
} aclrtFloatOverflowMode;

typedef enum {
    ACL_RT_STREAM_WORK_ADDR_PTR = 0, /**< pointer to model work addr */
    ACL_RT_STREAM_WORK_SIZE, /**< pointer to model work size */
    ACL_RT_STREAM_FLAG,
    ACL_RT_STREAM_PRIORITY,
} aclrtStreamConfigAttr;

typedef struct aclrtStreamConfigHandle {
    void* workptr;
    size_t workSize;
    size_t flag;
    uint32_t priority;
} aclrtStreamConfigHandle;

typedef struct aclrtUtilizationExtendInfo aclrtUtilizationExtendInfo;

typedef struct aclrtUtilizationInfo {
    int32_t cubeUtilization;
    int32_t vectorUtilization;
    int32_t aicpuUtilization;
    int32_t memoryUtilization;
    aclrtUtilizationExtendInfo *utilizationExtend; /**< reserved parameters, current version needs to be null */
} aclrtUtilizationInfo;

typedef struct tagRtGroupInfo aclrtGroupInfo;

typedef struct rtExceptionInfo aclrtExceptionInfo;

typedef enum aclrtMemLocationType {
    ACL_MEM_LOCATION_TYPE_HOST = 0, /**< reserved enum, current version not support */
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEM_LOCATION_TYPE_UNREGISTERED,
    ACL_MEM_LOCATION_TYPE_HOST_NUMA = 4, /*alloc host memeory via NUMA ID */
} aclrtMemLocationType;

typedef struct aclrtMemLocation {
    uint32_t id;
    aclrtMemLocationType type;
} aclrtMemLocation;

typedef struct aclrtPtrAttributes {
    aclrtMemLocation location;
    uint32_t pageSize;
    uint32_t rsv[4];
} aclrtPtrAttributes;

typedef struct aclrtMemUsageInfo {
    char name[32];
    uint64_t curMemSize;
    uint64_t memPeakSize;
    size_t reserved[8];
} aclrtMemUsageInfo;

typedef enum aclrtMemManagedRangeAttribute {
    ACL_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    ACL_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
    ACL_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
    ACL_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE,
    ACL_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID,
    ACL_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
    ACL_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE,
    ACL_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID,
} aclrtMemManagedRangeAttribute;

typedef enum aclrtMemAllocationType {
    ACL_MEM_ALLOCATION_TYPE_PINNED = 0,
} aclrtMemAllocationType;

typedef enum aclrtMemHandleType {
    ACL_MEM_HANDLE_TYPE_NONE = 0,
    ACL_MEM_HANDLE_TYPE_POSIX = 2,
} aclrtMemHandleType;

typedef enum aclrtMemSharedHandleType {
    ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT = 0x1,
    ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 0x2,
} aclrtMemSharedHandleType;

typedef struct aclrtMemFabricHandle { 
    uint8_t data[128];
} aclrtMemFabricHandle;

typedef struct aclrtPhysicalMemProp {
    aclrtMemHandleType handleType;
    aclrtMemAllocationType allocationType;
    aclrtMemAttr memAttr;
    aclrtMemLocation location;
    uint64_t reserve;
} aclrtPhysicalMemProp;

typedef enum aclrtMemGranularityOptions {
    ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
    ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    ACL_RT_MEM_ALLOC_GRANULARITY_UNDEF = 0xFFFF,
} aclrtMemGranularityOptions;

typedef void* aclrtDrvMemHandle;

typedef void (*aclrtCallback)(void *userData);

typedef void (*aclrtHostFunc)(void *args);

typedef void (*aclrtExceptionInfoCallback)(aclrtExceptionInfo *exceptionInfo);

typedef void (*aclrtOpExceptionCallback)(aclrtExceptionInfo *exceptionInfo, void *userData);

typedef enum aclrtDeviceStatus {
    ACL_RT_DEVICE_STATUS_NORMAL = 0,
    ACL_RT_DEVICE_STATUS_ABNORMAL,
    ACL_RT_DEVICE_STATUS_END = 0xFFFF,
} aclrtDeviceStatus;

typedef void* aclrtBinary;
typedef void* aclrtBinHandle;
typedef void* aclrtFuncHandle;
typedef void* aclrtArgsHandle;
typedef void* aclrtParamHandle;

typedef void *aclmdlRI;

typedef enum {
    ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0,
    ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL,
    ACL_MODEL_RI_CAPTURE_MODE_RELAXED,
} aclmdlRICaptureMode;

typedef enum {
    ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0,
    ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
    ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
} aclmdlRICaptureStatus;

#define MAX_MEM_UCE_INFO_ARRAY_SIZE 128
#define UCE_INFO_RESERVED_SIZE 14

typedef struct aclrtMemUceInfo {
    void* addr;
    size_t len;
    size_t reserved[UCE_INFO_RESERVED_SIZE];
} aclrtMemUceInfo;

typedef enum {
    ACL_RT_NO_ERROR = 0,
    ACL_RT_ERROR_MEMORY = 1,
    ACL_RT_ERROR_L2 = 2,
    ACL_RT_ERROR_AICORE = 3,
    ACL_RT_ERROR_LINK = 4,
    ACL_RT_ERROR_OTHERS = 0xFFFF,
} aclrtErrorType;

typedef enum aclrtAicoreErrorType {
    ACL_RT_AICORE_ERROR_UNKNOWN,
    ACL_RT_AICORE_ERROR_SW,
    ACL_RT_AICORE_ERROR_HW_LOCAL,
} aclrtAicoreErrorType;

#define ACL_RT_MEM_UCE_INFO_MAX_NUM 20
typedef struct {
    size_t arraySize;
    aclrtMemUceInfo memUceInfoArray[ACL_RT_MEM_UCE_INFO_MAX_NUM];
} aclrtMemUceInfoArray;

typedef union aclrtErrorInfoDetail {
    aclrtMemUceInfoArray uceInfo;
    aclrtAicoreErrorType aicoreErrType;
} aclrtErrorInfoDetail;

typedef struct aclrtErrorInfo {
    uint8_t tryRepair;
    uint8_t hasDetail;
    uint8_t reserved[2];
    aclrtErrorType errorType;
    aclrtErrorInfoDetail detail;
} aclrtErrorInfo;

typedef enum aclrtCmoType {
    ACL_RT_CMO_TYPE_PREFETCH = 0,
    ACL_RT_CMO_TYPE_WRITEBACK,
    ACL_RT_CMO_TYPE_INVALID,
    ACL_RT_CMO_TYPE_FLUSH,
} aclrtCmoType;

typedef enum aclrtLastErrLevel {
    ACL_RT_THREAD_LEVEL = 0,
} aclrtLastErrLevel;

#define ACL_RT_BINARY_MAGIC_ELF_AICORE      0x43554245U
#define ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE 0x41415246U
#define ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE   0x41494343U

typedef enum aclrtBinaryLoadOptionType {
    ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD = 1,
    ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC = 2,
    ACL_RT_BINARY_LOAD_OPT_MAGIC = 2,
    ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE = 3,
} aclrtBinaryLoadOptionType;

typedef union aclrtBinaryLoadOptionValue {
    uint32_t isLazyLoad;
    uint32_t magic;
    int32_t cpuKernelMode;
    uint32_t rsv[4];
} aclrtBinaryLoadOptionValue;

typedef struct {
    aclrtBinaryLoadOptionType type;
    aclrtBinaryLoadOptionValue value;
} aclrtBinaryLoadOption;

typedef struct aclrtBinaryLoadOptions {
    aclrtBinaryLoadOption *options;
    size_t numOpt;
} aclrtBinaryLoadOptions;

typedef enum {
    ACL_RT_ENGINE_TYPE_AIC = 0,
    ACL_RT_ENGINE_TYPE_AIV,
} aclrtEngineType;

typedef enum aclrtLaunchKernelAttrId {
    ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 1,
    ACL_RT_LAUNCH_KERNEL_ATTR_LOCAL_MEMORY_SIZE 
        ACL_DEPRECATED_MESSAGE("Use ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE instead") = 2, // DEPRECATED: Use ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE
    ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE = 2,
    ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE = 3,
    ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET = 4,
    ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH = 5,
    ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP = 6,
    ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT = 7,
    // ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT and ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US cannot be carried at the same time
    ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US = 8,
} aclrtLaunchKernelAttrId;

/**
 * @ingroup rts_kernel
 * @brief kernel launch option timeout value
 */
typedef struct {
    uint32_t timeoutLow;  // low  32bit
    uint32_t timeoutHigh; // high 32bit
} aclrtTimeoutUs;

typedef union aclrtLaunchKernelAttrValue {
    uint8_t schemMode;
    ACL_DEPRECATED_MESSAGE("Use dynUbufSize instead")
    uint32_t localMemorySize; // DEPRECATED: Use dynUBufSize
    uint32_t dynUBufSize;
    aclrtEngineType engineType;
    uint32_t blockDimOffset;
    uint8_t isBlockTaskPrefetch;
    uint8_t isDataDump;
    uint16_t timeout;   // unit: s
    aclrtTimeoutUs timeoutUs; // unit: us
    uint32_t rsv[4];
} aclrtLaunchKernelAttrValue;

typedef struct aclrtLaunchKernelAttr {
    aclrtLaunchKernelAttrId id;
    aclrtLaunchKernelAttrValue value;
} aclrtLaunchKernelAttr;

typedef struct aclrtLaunchKernelCfg {
    aclrtLaunchKernelAttr *attrs;
    size_t numAttrs;
} aclrtLaunchKernelCfg;

typedef enum {
    ACL_STREAM_ATTR_FAILURE_MODE         = 1,
    ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK = 2,
    ACL_STREAM_ATTR_USER_CUSTOM_TAG      = 3,
    ACL_STREAM_ATTR_CACHE_OP_INFO        = 4,
} aclrtStreamAttr;

typedef union {
    uint64_t failureMode;
    uint32_t overflowSwitch;
    uint32_t userCustomTag;
    uint32_t cacheOpInfoSwitch;
    uint32_t reserve[4];
} aclrtStreamAttrValue;

typedef enum aclmdlRITaskType {
    ACL_MODEL_RI_TASK_DEFAULT,
    ACL_MODEL_RI_TASK_KERNEL,
    ACL_MODEL_RI_TASK_EVENT_RECORD,
    ACL_MODEL_RI_TASK_EVENT_WAIT,
    ACL_MODEL_RI_TASK_EVENT_RESET,
    ACL_MODEL_RI_TASK_VALUE_WRITE,
    ACL_MODEL_RI_TASK_VALUE_WAIT,
} aclmdlRITaskType;

typedef struct aclmdlRIKernelTaskParams {
    aclrtFuncHandle funcHandle;
    aclrtLaunchKernelCfg* cfg;
    void* args;
    uint32_t isHostArgs;
    size_t argsSize;
    uint32_t numBlocks;
    uint32_t rsv[10];
} aclmdlRIKernelTaskParams;

typedef struct aclmdlRIEventRecordTaskParams {
    aclrtEvent event;
    uint64_t eventFlag;
} aclmdlRIEventRecordTaskParams;

typedef struct aclmdlRIEventWaitTaskParams {
    aclrtEvent event;
    uint64_t eventFlag;
} aclmdlRIEventWaitTaskParams;

typedef struct aclmdlRIEventResetTaskParams {
    aclrtEvent event;
    uint64_t eventFlag;
} aclmdlRIEventResetTaskParams;

typedef struct aclmdlRIValueWriteTaskParams {
    void* devAddr;
    uint64_t value;
} aclmdlRIValueWriteTaskParams;

typedef struct aclmdlRIValueWaitTaskParams {
    void* devAddr;
    uint64_t value;
    uint32_t flag;
} aclmdlRIValueWaitTaskParams;

typedef struct aclmdlRITaskParams {
    aclmdlRITaskType type;
    uint32_t rsv0[3];
    aclrtTaskGrp taskGrp;
    void* opInfoPtr;
    size_t opInfoSize;
    uint8_t rsv1[32];

    union {
        uint8_t rsv2[128];
        struct aclmdlRIKernelTaskParams kernelTaskParams;
        struct aclmdlRIEventRecordTaskParams eventRecordTaskParams;
        struct aclmdlRIEventWaitTaskParams eventWaitTaskParams;
        struct aclmdlRIEventResetTaskParams eventResetTaskParams;
        struct aclmdlRIValueWriteTaskParams valueWriteTaskParams;
        struct aclmdlRIValueWaitTaskParams valueWaitTaskParams;
    };
} aclmdlRITaskParams;

typedef enum {
    ACL_DEV_ATTR_AICPU_CORE_NUM  = 1,    // number of AI CPUs

    ACL_DEV_ATTR_AICORE_CORE_NUM = 101,  // number of AI Cores
    ACL_DEV_ATTR_CUBE_CORE_NUM   = 102,  // number of Cube Cores

    ACL_DEV_ATTR_VECTOR_CORE_NUM = 201,  // number of Vector Cores
    ACL_DEV_ATTR_WARP_SIZE       = 202,  // number of threads in a Warp
    ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE = 203,    // maximum number of concurrent threads per Vector Core
    ACL_DEV_ATTR_LOCAL_MEM_PER_VECTOR_CORE
        ACL_DEPRECATED_MESSAGE("Use ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE instead") = 204,    // DEPRECATED: Use ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE
    ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE  = 204,    // maximum available local memory per Vector Core, in Bytes

    ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE = 301,    // total available global memory on the Device, in Bytes
    ACL_DEV_ATTR_L2_CACHE_SIZE         = 302,    // L2 Cache size, in Bytes

    ACL_DEV_ATTR_SMP_ID = 401U,                 // indicates whether devices are on the same OS
    ACL_DEV_ATTR_PHY_CHIP_ID = 402U,            // physical chip id
    ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID = 403U,    // DEPRECATED: Use ACL_DEV_ATTR_SUPER_POD_DEVICE_ID
    ACL_DEV_ATTR_SUPER_POD_DEVICE_ID = 403U,    // super pod device id
    ACL_DEV_ATTR_SUPER_POD_SERVER_ID = 404U,    // super pod server id
    ACL_DEV_ATTR_SUPER_POD_ID = 405U,           // super pod id
    ACL_DEV_ATTR_CUST_OP_PRIVILEGE = 406U,      // indicates whether the custom operator privilege is enabled
    ACL_DEV_ATTR_MAINBOARD_ID = 407U,           // mainborad id

    ACL_DEV_ATTR_IS_VIRTUAL = 501U,             // whether it is in compute power splitting mode
} aclrtDevAttr;

typedef enum {
    ACL_FEATURE_TSCPU_TASK_UPDATE_SUPPORT_AIC_AIV = 1,
    ACL_FEATURE_SYSTEM_MEMQ_EVENT_CROSS_DEV       = 21,
} aclrtDevFeatureType;

typedef enum {
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM   = 10,
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX   = 11,
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN   = 12,
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_EQUAL = 13,
} aclrtReduceKind;

typedef enum {
    ACL_RT_DEV_RES_CUBE_CORE = 0,
    ACL_RT_DEV_RES_VECTOR_CORE,
} aclrtDevResLimitType;

typedef enum {
    ACL_RT_EQUAL = 0,
    ACL_RT_NOT_EQUAL,
    ACL_RT_GREATER,
    ACL_RT_GREATER_OR_EQUAL,
    ACL_RT_LESS,
    ACL_RT_LESS_OR_EQUAL
} aclrtCondition;

typedef enum {
    ACL_RT_SWITCH_INT32 = 0,
    ACL_RT_SWITCH_INT64 = 1,
} aclrtCompareDataType;

typedef struct {
    aclrtCmoType cmoType;
    uint32_t barrierId;
} aclrtBarrierCmoInfo;

#define ACL_RT_CMO_MAX_BARRIER_NUM 6U

typedef struct {
    size_t barrierNum;
    aclrtBarrierCmoInfo cmoInfo[ACL_RT_CMO_MAX_BARRIER_NUM];
} aclrtBarrierTaskInfo;

#define ACL_RT_DEVS_TOPOLOGY_HCCS     0x01ULL
#define ACL_RT_DEVS_TOPOLOGY_PIX      0x02ULL
#define ACL_RT_DEVS_TOPOLOGY_PIB      0x04ULL
#define ACL_RT_DEVS_TOPOLOGY_PHB      0x08ULL
#define ACL_RT_DEVS_TOPOLOGY_SYS      0x10ULL
#define ACL_RT_DEVS_TOPOLOGY_SIO      0x20ULL
#define ACL_RT_DEVS_TOPOLOGY_HCCS_SW  0x40ULL

typedef struct {
    aclrtMemLocation dstLoc;
    aclrtMemLocation srcLoc;
    uint8_t rsv[16];
} aclrtMemcpyBatchAttr;

typedef struct {
    uint32_t addrOffset;
    uint32_t dataOffset;
} aclrtPlaceHolderInfo;

typedef struct {
    uint8_t isAddr;
    uint8_t valueOrAddr[8];
    uint8_t size;
    uint8_t rsv[6];
} aclrtRandomParaInfo;

// dropout bitmask
typedef struct {
    aclrtRandomParaInfo dropoutRation;
} aclrtDropoutBitmaskInfo;

// uniform distribution
typedef struct {
    aclrtRandomParaInfo min;
    aclrtRandomParaInfo max;
} aclrtUniformDisInfo;

// normal distribution
typedef struct {
    aclrtRandomParaInfo mean;
    aclrtRandomParaInfo stddev;
} aclrtNormalDisInfo;

typedef enum {
    ACL_RT_RANDOM_NUM_FUNC_TYPE_DROPOUT_BITMASK = 0, // dropout bitmask
    ACL_RT_RANDOM_NUM_FUNC_TYPE_UNIFORM_DIS, // uniform distribution
    ACL_RT_RANDOM_NUM_FUNC_TYPE_NORMAL_DIS, // normal distribution
    ACL_RT_RANDOM_NUM_FUNC_TYPE_TRUNCATED_NORMAL_DIS, // truncated normal distribution
} aclrtRandomNumFuncType;

typedef struct {
    aclrtRandomNumFuncType funcType;
    union {
        aclrtDropoutBitmaskInfo dropoutBitmaskInfo;
        aclrtUniformDisInfo uniformDisInfo;
        aclrtNormalDisInfo normalDisInfo;
    } paramInfo;
} aclrtRandomNumFuncParaInfo;

typedef struct {
    aclDataType dataType;
    aclrtRandomNumFuncParaInfo randomNumFuncParaInfo;
    void *randomParaAddr;
    void *randomResultAddr;
    void *randomCounterAddr;
    aclrtRandomParaInfo randomSeed;
    aclrtRandomParaInfo randomNum;
    uint8_t rsv[8];
} aclrtRandomNumTaskInfo;

typedef enum {
    ACL_RT_DEVICE_TASK_ABORT_PRE = 0,
    ACL_RT_DEVICE_TASK_ABORT_POST,
} aclrtDeviceTaskAbortStage;

typedef enum {
    ACL_RT_UPDATE_RANDOM_TASK = 1,
    ACL_RT_UPDATE_AIC_AIV_TASK,
} aclrtUpdateTaskAttrId;

typedef struct {
    void *srcAddr;
    size_t size;
    uint32_t rsv[4];
} aclrtRandomTaskUpdateAttr;

typedef struct {
    void *binHandle; // program handle
    void *funcEntryAddr;
    void *blockDimAddr;
    uint32_t rsv[4];
} aclrtAicAivTaskUpdateAttr;

typedef union {
    aclrtRandomTaskUpdateAttr randomTaskAttr;
    aclrtAicAivTaskUpdateAttr aicAivTaskAttr;
} aclrtUpdateTaskAttrVal;

typedef struct {
    aclrtUpdateTaskAttrId id;
    aclrtUpdateTaskAttrVal val;
} aclrtTaskUpdateInfo;

typedef enum {
    ACL_RT_DEVICE_STATE_SET_PRE = 0,
    ACL_RT_DEVICE_STATE_SET_POST,
    ACL_RT_DEVICE_STATE_RESET_PRE,
    ACL_RT_DEVICE_STATE_RESET_POST,
} aclrtDeviceState;

typedef void (*aclrtDeviceStateCallback)(int32_t deviceId, aclrtDeviceState state, void *args);

/*
 * BackUp Flow:
 * LOCK_PRE → [aclrtSnapShotProcessLock] → BACKUP_PRE → [aclrtSnapShotProcessBackup] → BACKUP_POST → [aclrtSnapShotProcessUnlock] -> UNLOCK_POST
 * Restore Flow:
 * RESTORE_PRE → [aclrtSnapShotProcessRestore] → RESTORE_POST → [aclrtSnapShotProcessUnlock] → UNLOCK_POST
 */
typedef enum {
    ACL_RT_SNAPSHOT_LOCK_PRE = 0,
    ACL_RT_SNAPSHOT_BACKUP_PRE,
    ACL_RT_SNAPSHOT_BACKUP_POST,
    ACL_RT_SNAPSHOT_RESTORE_PRE,
    ACL_RT_SNAPSHOT_RESTORE_POST,
    ACL_RT_SNAPSHOT_UNLOCK_POST,
} aclrtSnapShotStage;

typedef uint32_t (*aclrtSnapShotCallBack)(int32_t deviceId, void* args);

typedef struct aclrtUuid {
    char bytes[16];
} aclrtUuid;

typedef enum {
    ACL_RT_STREAM_STATE_CREATE_POST = 1,
    ACL_RT_STREAM_STATE_DESTROY_PRE,
} aclrtStreamState;

typedef void (*aclrtStreamStateCallback)(aclrtStream stm, aclrtStreamState state, void *args);

typedef int32_t (*aclrtDeviceTaskAbortCallback)(int32_t deviceId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void *args);

typedef enum {
    ACL_FUNC_ATTR_KERNEL_TYPE = 1,
    ACL_FUNC_ATTR_KERNEL_RATIO = 2,
    ACL_FUNC_ATTR_KERNEL_SCHED_MODE = 3,
} aclrtFuncAttribute;

typedef enum {
    ACL_KERNEL_TYPE_AICORE = 0, // MIX KERNEL
    ACL_KERNEL_TYPE_CUBE = 1,   // AI CUBE CORE
    ACL_KERNEL_TYPE_VECTOR = 2, // AI VECTOR CORE
    ACL_KERNEL_TYPE_MIX = 3,
    ACL_KERNEL_TYPE_AICPU = 100,
} aclrtKernelType;

#define ACL_RT_MEM_TYPE_DEV   (0X2U)
#define ACL_RT_MEM_TYPE_DVPP  (0X8U)
#define ACL_RT_MEM_TYPE_RSVD  (0X10U)

typedef enum {
    ACL_RT_CNT_NOTIFY_RECORD_SET_VALUE_MODE = 0,
    ACL_RT_CNT_NOTIFY_RECORD_ADD_MODE = 1,
    ACL_RT_CNT_NOTIFY_RECORD_BIT_OR_MODE = 2,

    ACL_RT_CNT_NOTIFY_RECORD_BIT_AND_MODE = 4,
} aclrtCntNotifyRecordMode;

typedef struct {
    aclrtCntNotifyRecordMode mode;
    uint32_t value;
} aclrtCntNotifyRecordInfo;

typedef enum {
    ACL_RT_CNT_NOTIFY_WAIT_LESS_MODE = 0,
    ACL_RT_CNT_NOTIFY_WAIT_EQUAL_MODE = 1,
    ACL_RT_CNT_NOTIFY_WAIT_BIGGER_MODE = 2,
    ACL_RT_CNT_NOTIFY_WAIT_BIGGER_OR_EQUAL_MODE = 3,
    ACL_RT_CNT_NOTIFY_WAIT_EQUAL_WITH_BITMASK_MODE = 4,
} aclrtCntNotifyWaitMode;

typedef struct {
    aclrtCntNotifyWaitMode mode;
    uint32_t value;
    uint32_t timeout;
    uint8_t isClear;
    uint8_t rsv[3];
} aclrtCntNotifyWaitInfo;

typedef enum {
    ACL_RT_MEM_ACCESS_FLAGS_NONE = 0x0,
    ACL_RT_MEM_ACCESS_FLAGS_READ = 0x1,
    ACL_RT_MEM_ACCESS_FLAGS_READWRITE = 0x3,
} aclrtMemAccessFlags;

typedef struct {
    aclrtMemAccessFlags flags;
    aclrtMemLocation location;
    uint8_t rsv[12];
} aclrtMemAccessDesc;

#define ACL_PKG_VERSION_MAX_SIZE       128
#define ACL_PKG_VERSION_PARTS_MAX_SIZE 64

/**
 * @ingroup AscendCL
 * @brief enum for CANN package name
 */
typedef enum aclCANNPackageName {
    ACL_PKG_NAME_CANN,
    ACL_PKG_NAME_RUNTIME,
    ACL_PKG_NAME_COMPILER,
    ACL_PKG_NAME_HCCL,
    ACL_PKG_NAME_TOOLKIT,
    ACL_PKG_NAME_OPP,
    ACL_PKG_NAME_OPP_KERNEL,
    ACL_PKG_NAME_DRIVER
} aclCANNPackageName;

/**
 * @ingroup AscendCL
 * @brief struct for storaging CANN package version
 */
typedef struct aclCANNPackageVersion {
    char version[ACL_PKG_VERSION_MAX_SIZE];
    char majorVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char minorVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char releaseVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char patchVersion[ACL_PKG_VERSION_PARTS_MAX_SIZE];
    char reserved[ACL_PKG_VERSION_MAX_SIZE];
} aclCANNPackageVersion;

typedef enum {
    ACL_RT_HAC_TYPE_STARS = 0,
    ACL_RT_HAC_TYPE_AICPU,
    ACL_RT_HAC_TYPE_AIC,
    ACL_RT_HAC_TYPE_AIV,
    ACL_RT_HAC_TYPE_PCIEDMA,
    ACL_RT_HAC_TYPE_RDMA,
    ACL_RT_HAC_TYPE_SDMA,
    ACL_RT_HAC_TYPE_DVPP,
    ACL_RT_HAC_TYPE_UDMA,
    ACL_RT_HAC_TYPE_CCU  
} aclrtHacType;

typedef enum {
    ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED = 0,
    ACL_RT_HOST_MEM_MAP_SUPPORTED
} aclrtHostMemMapCapability;

typedef struct aclrtIpcEventHandle {
    char reserved[ACL_IPC_EVENT_HANDLE_SIZE];
} aclrtIpcEventHandle;

struct MemAttrMapping {
    uint32_t pgType;
    uint32_t memType;
    bool isHostAlloc;
    aclrtMemAttr memAttr;
};

typedef enum aclrtAtomicOperationCapability {
    ACL_RT_ATOMIC_CAPABILITY_SIGNED = 1U << 0,
    ACL_RT_ATOMIC_CAPABILITY_UNSIGNED = 1U << 1,
    ACL_RT_ATOMIC_CAPABILITY_REDUCATION = 1U << 2,
    ACL_RT_ATOMIC_CAPABILITY_SCALAR8 = 1U << 3,
    ACL_RT_ATOMIC_CAPABILITY_SCALAR16 = 1U << 4,
    ACL_RT_ATOMIC_CAPABILITY_SCALAR32 = 1U << 5,
    ACL_RT_ATOMIC_CAPABILITY_SCALAR64 = 1U << 6,
    ACL_RT_ATOMIC_CAPABILITY_SCALAR128 = 1U << 7,
    ACL_RT_ATOMIC_CAPABILITY_VECTOR32X4 = 1U << 8,
} aclrtAtomicOperationCapability;

typedef enum aclrtAtomicOperation {
    ACL_RT_ATOMIC_OPERATION_INTEGER_ADD = 0,
    ACL_RT_ATOMIC_OPERATION_INTEGER_MIN = 1,
    ACL_RT_ATOMIC_OPERATION_INTEGER_MAX = 2,
    ACL_RT_ATOMIC_OPERATION_INTEGER_INCREMENT = 3,
    ACL_RT_ATOMIC_OPERATION_INTEGER_DECREMENT = 4,
    ACL_RT_ATOMIC_OPERATION_AND = 5,
    ACL_RT_ATOMIC_OPERATION_OR = 6,
    ACL_RT_ATOMIC_OPERATION_XOR = 7,
    ACL_RT_ATOMIC_OPERATION_EXCHANGE = 8,
    ACL_RT_ATOMIC_OPERATION_CAS = 9,
    ACL_RT_ATOMIC_OPERATION_FLOAT_ADD = 10,
    ACL_RT_ATOMIC_OPERATION_FLOAT_MIN = 11,
    ACL_RT_ATOMIC_OPERATION_FLOAT_MAX = 12,

    ACL_RT_ATOMIC_OPERATION_DMA_ADD = 30,
    ACL_RT_ATOMIC_OPERATION_DMA_MIN = 31,
    ACL_RT_ATOMIC_OPERATION_DMA_MAX = 32,

    ACL_RT_ATOMIC_OPERATION_SIMD_SCALAR_ADD = 40,
    ACL_RT_ATOMIC_OPERATION_SIMD_SCALAR_MIN = 41,
    ACL_RT_ATOMIC_OPERATION_SIMD_SCALAR_MAX = 42,
    ACL_RT_ATOMIC_OPERATION_SIMD_SCALAR_CAS = 43,
    ACL_RT_ATOMIC_OPERATION_SIMD_SCALAR_EXCH = 44,
} aclrtAtomicOperation;

/**
 * @ingroup AscendCL
 * @brief acl initialize
 *
 * @par Restriction
 * The aclInit interface can be called only once in a process
 * @param configPath [IN]    the config path,it can be NULL
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclInit(const char *configPath);

/**
 * @ingroup AscendCL
 * @brief acl finalize
 *
 * @par Restriction
 * Need to call aclFinalize before the process exits.
 * After calling aclFinalize,the services cannot continue to be used normally.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalize();

/**
 * @ingroup AscendCL
 * @brief acl finalize reference
 *
 * @par Restriction
 * This interface decrements the internal reference count each time it is called.
 * Resources are only released when the reference count reaches 0.
 * To get the current reference count, pass a valid pointer to refCount.
 * To ignore the reference count, pass nullptr instead.
 *
 * @param refCount [IN/OUT] Pointer to receive current reference count after calling aclFinalizeReference; can be nullptr.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalizeReference(uint64_t *refCount);

/**
 * @ingroup AscendCL
 * @brief get recent error message
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY const char *aclGetRecentErrMsg();

/**
 * @ingroup AscendCL
 * @brief query CANN package version
 *
 * @param name[IN] CANN package name
 * @param version[OUT] CANN package version information
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_DEPRECATED_MESSAGE("aclsysGetCANNVersion is deprecated, use aclsysGetVersionStr and aclsysGetVersionNum instead")
ACL_FUNC_VISIBILITY aclError aclsysGetCANNVersion(aclCANNPackageName name, aclCANNPackageVersion *version);

/**
 * @ingroup AscendCL
 * @brief Query the CANN package version based on the package name and return it as a string.
 *
 * @param pkgName[IN] CANN package name
 * @param versionStr[OUT] CANN package version number in string format
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclsysGetVersionStr(char *pkgName, char *versionStr);

/**
 * @ingroup AscendCL
 * @brief Query the CANN package version based on the package name and return it as a number.
 *
 * @param pkgName[IN] CANN package name
 * @param versionNum[OUT] CANN package version number
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclsysGetVersionNum(char *pkgName, int32_t *versionNum);

/**
 * @ingroup AscendCL
 * @brief peek at last error by level
 *
 * @param level [IN] error level
 *
 * @retval Runtime error code
 */
ACL_FUNC_VISIBILITY aclError aclrtPeekAtLastError(aclrtLastErrLevel level);

/**
 * @ingroup AscendCL
 * @brief get last error by level
 *
 * @param level [IN] error level
 *
 * @retval Runtime error code
 */
ACL_FUNC_VISIBILITY aclError aclrtGetLastError(aclrtLastErrLevel level);


/**
 * @ingroup AscendCL
 * @brief Set a callback function to handle exception information
 *
 * @param callback [IN] callback function to handle exception information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback);

/**
 * @ingroup AscendCL
 * @brief Get task id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The task id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get stream id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The stream id from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get thread id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetThreadIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get device id from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The thread id of fail task
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get error code from exception information
 *
 * @param info [IN]   pointer of exception information
 *
 * @retval The error code from exception information
 * @retval 0xFFFFFFFF if info is null
 */
ACL_FUNC_VISIBILITY uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info);

/**
 * @ingroup AscendCL
 * @brief Get args from exception information
 *
 * @param info [IN]   pointer of exception information
 * @param info [OUT]   dev args of exception information
 * @param info [OUT]   dev args len of exception information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetArgsFromExceptionInfo(const aclrtExceptionInfo *info, void **devArgsPtr, uint32_t *devArgsLen);

/**
 * @ingroup AscendCL
 * @brief Get func handle from exception information
 *
 * @param info [IN]   pointer of exception information
 * @param info [OUT]   kernel func of exception information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetFuncHandleFromExceptionInfo(const aclrtExceptionInfo *info, aclrtFuncHandle *func);

/**
 * @ingroup AscendCL
 * @brief Set exception information callback handle to binHandle
 *
 * @param info [IN]   binary bin handle
 * @param info [IN]   exception callback of binary bin handle
 * @param info [IN]   exception userData of binary bin handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinarySetExceptionCallback(aclrtBinHandle binHandle, aclrtOpExceptionCallback callback, void *userData);

/**
 * @ingroup AscendCL
 * @brief The thread that handles the callback function on the Stream
 *
 * @param threadId [IN] thread ID
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Add a callback function to be executed on the host
 *        to the task queue of the Stream
 *
 * @param fn [IN]   Specify the callback function to be added
 *                  The function prototype of the callback function is:
 *                  typedef void (*aclrtCallback)(void *userData);
 * @param userData [IN]   User data to be passed to the callback function
 * @param blockType [IN]  callback block type
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType,
                                                 aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief After waiting for a specified time, trigger callback processing
 *
 * @par Function
 *  The thread processing callback specified by
 *  the aclrtSubscribeReport interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSubscribeReport
 */
ACL_FUNC_VISIBILITY aclError aclrtProcessReport(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Cancel thread registration,
 *        the callback function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param threadId [IN]   thread ID
 * @param stream [IN]     stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create context and associates it with the calling thread
 *
 * @par Function
 * The following use cases are supported:
 * @li If you don't call the aclrtCreateContext interface
 * to explicitly create the context,
 * the system will use the default context, which is implicitly created
 * when the aclrtSetDevice interface is called.
 * @li If multiple contexts are created in a process
 * (there is no limit on the number of contexts),
 * the current thread can only use one of them at the same time.
 * It is recommended to explicitly specify the context of the current thread
 * through the aclrtSetCurrentContext interface to increase.
 * the maintainability of the program.
 *
 * @param  context [OUT]    point to the created context
 * @param  deviceId [IN]    device to create context on
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetDevice | aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief destroy context instance
 *
 * @par Function
 * Can only destroy context created through aclrtCreateContext interface
 *
 * @param  context [IN]   the context to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief set the context of the thread
 *
 * @par Function
 * The following scenarios are supported:
 * @li If the aclrtCreateContext interface is called in a thread to explicitly
 * create a Context (for example: ctx1), the thread's Context can be specified
 * without calling the aclrtSetCurrentContext interface.
 * The system uses ctx1 as the context of thread1 by default.
 * @li If the aclrtCreateContext interface is not explicitly created,
 * the system uses the default context as the context of the thread.
 * At this time, the aclrtDestroyContext interface cannot be used to release
 * the default context.
 * @li If the aclrtSetCurrentContext interface is called multiple times to
 * set the thread's Context, the last one prevails.
 *
 * @par Restriction
 * @li If the cevice corresponding to the context set for the thread
 * has been reset, you cannot set the context as the context of the thread,
 * otherwise a business exception will result.
 * @li It is recommended to use the context created in a thread.
 * If the aclrtCreateContext interface is called in thread A to create a context,
 * and the context is used in thread B,
 * the user must guarantee the execution order of tasks in the same stream
 * under the same context in two threads.
 *
 * @param  context [IN]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateContext | aclrtDestroyContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context);

/**
 * @ingroup AscendCL
 * @brief get the context of the thread
 *
 * @par Function
 * If the user calls the aclrtSetCurrentContext interface
 * multiple times to set the context of the current thread,
 * then the last set context is obtained
 *
 * @param  context [OUT]   the current context of the thread
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetCurrentContext
 */
ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context);

/**
 * @ingroup AscendCL
 * @brief get system param option value in current context
 *
 * @param opt[IN] system option
 * @param value[OUT] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t *value);

/**
 * @ingroup AscendCL
 * @brief set system param option value in current context
 *
 * @param opt[IN] system option
 * @param value[IN] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value);

/**
 * @ingroup AscendCL
 * @brief get system param option value in current process
 *
 * @param opt[IN] system option
 * @param value[OUT] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t *value);

/**
 * @ingroup AscendCL
 * @brief set system param option value in current process
 *
 * @param opt[IN] system option
 * @param value[IN] value of system option
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value);

/**
 * @ingroup AscendCL
 * @brief Specify the device to use for the operation
 * implicitly create the default context and the default stream
 *
 * @par Function
 * The following use cases are supported:
 * @li Device can be specified in the process or thread.
 * If you call the aclrtSetDevice interface multiple
 * times to specify the same device,
 * you only need to call the aclrtResetDevice interface to reset the device.
 * @li The same device can be specified for operation
 *  in different processes or threads.
 * @li Device is specified in a process,
 * and multiple threads in the process can share this device to explicitly
 * create a Context (aclrtCreateContext interface).
 * @li In multi-device scenarios, you can switch to other devices
 * through the aclrtSetDevice interface in the process.
 *
 * @param  deviceId [IN]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtResetDevice |aclrtCreateContext
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Reset the current operating Device and free resources on the device,
 * including the default context, the default stream,
 * and all streams created under the default context,
 * and synchronizes the interface.
 * If the task under the default context or stream has not been completed,
 * the system will wait for the task to complete before releasing it.
 *
 * @par Restriction
 * @li The Context, Stream, and Event that are explicitly created
 * on the device to be reset. Before resetting,
 * it is recommended to follow the following interface calling sequence,
 * otherwise business abnormalities may be caused.
 * @li Interface calling sequence:
 * call aclrtDestroyEvent interface to release Event or
 * call aclrtDestroyStream interface to release explicitly created Stream->
 * call aclrtDestroyContext to release explicitly created Context->
 * call aclrtResetDevice interface
 *
 * @param  deviceId [IN]   the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Reset the current operating Device and free resources on the device by FORCE,
 * including the default context, the default stream,
 * and all streams created under the default context,
 * and synchronizes the interface.
 * If the task under the default context or stream has not been completed,
 * the system will wait for the task to complete before releasing it.
 * No matter how many times you call aclrtSetDevice for the same device id,
 * you only need to call aclrtResetDeviceForce once for resetting.
 *
 * @par Restriction
 * @li The Context, Stream, and Event that are explicitly created
 * on the device to be reset. Before resetting,
 * it is recommended to follow the following interface calling sequence,
 * otherwise business abnormalities may be caused.
 * @li Interface calling sequence:
 * call aclrtDestroyEvent interface to release Event or
 * call aclrtDestroyStream interface to release explicitly created Stream->
 * call aclrtDestroyContext to release explicitly created Context->
 * call aclrtResetDeviceForce interface
 *
 * @param  deviceId [IN]   the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 * 
 * @see aclrtResetDevice
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDeviceForce(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief get target device of current thread
 *
 * @param deviceId [OUT]  the device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDevice(int32_t *deviceId);

/**
 * @ingroup AscendCL
 * @brief set stream failure mode
 *
 * @param stream [IN]  the stream to set
 * @param mode [IN]  stream failure mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode);

/**
 * @ingroup AscendCL
 * @brief get target side
 *
 * @param runMode [OUT]    the run mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetRunMode(aclrtRunMode *runMode);

/**
 * @ingroup AscendCL
 * @brief Wait for compute device to finish
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDevice(void);

/**
 * @ingroup AscendCL
 * @brief Wait for compute device to finish and set timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeDeviceWithTimeout(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Set Scheduling TS
 *
 * @param tsId [IN]   the ts id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetTsDevice(aclrtTsId tsId);

/**
 * @ingroup AscendCL
 * @brief Query the comprehensive usage rate of device
 * @param deviceId [IN] the need query's deviceId
 * @param utilizationInfo [OUT] the usage rate of device
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo);

/**
 * @ingroup AscendCL
 * @brief get total device number.
 *
 * @param count [OUT]    the device number
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEvent(aclrtEvent *event);

/**
 * @ingroup AscendCL
 * @brief create event instance with flag
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief create event instance with flag, event can be reused naturally
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy event instance
 *
 * @par Function
 *  Only events created through the aclrtCreateEvent interface can be
 *  destroyed, synchronous interfaces. When destroying an event,
 *  the user must ensure that the tasks involved in the aclrtSynchronizeEvent
 *  interface or the aclrtStreamWaitEvent interface are completed before
 *  they are destroyed.
 *
 * @param  event [IN]   event to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtSynchronizeEvent | aclrtStreamWaitEvent
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Record an Event in the Stream
 *
 * @param event [IN]    event to record
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Reset an event
 *
 * @par Function
 *  Users need to make sure to wait for the tasks in the Stream
 *  to complete before resetting the Event
 *
 * @param event [IN]    event to reset
 * @param stream [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream);

 /**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclrtQueryEvent is deprecated, use aclrtQueryEventStatus instead")
ACL_FUNC_VISIBILITY aclError aclrtQueryEvent(aclrtEvent event, aclrtEventStatus *status);

/**
 * @ingroup AscendCL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event recorded status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status);

/**
* @ingroup AscendCL
* @brief Queries an event's wait-status
*
* @param  event [IN]    event to query
* @param  status [OUT]  event wait-status
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *status);

/**
 * @ingroup AscendCL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 * -1 means waiting indefinitely, 0 means check whether synchronization is immediately completed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief computes the elapsed time between events.
 *
 * @param ms [OUT]     time between start and end in ms
 * @param start [IN]   starting event
 * @param end [IN]     ending event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtRecordEvent | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent);

/**
 * @ingroup AscendCL
 * @brief get syscnt when event recorded.
 *
 * @param event [IN]          event to be record
 * @param timestamp [OUT]     syscnt
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateEvent | aclrtRecordEvent | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtEventGetTimestamp(aclrtEvent event, uint64_t *timestamp);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device, real alloc size is aligned to 32 bytes and padded with 32 bytes
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMalloc interface needs to be released
 * through the aclrtFree interface.
 * @li Before calling the media data processing interface,
 * if you need to apply memory on the device to store input or output data,
 * you need to call acldvppMalloc to apply for memory.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | acldvppMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr,
                                         size_t size,
                                         aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief alloc memory on device, real alloc size is aligned to 32 bytes with no padding
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocAlign32 interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc | aclrtMallocCached
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocAlign32(void **devPtr,
                                                size_t size,
                                                aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief allocate memory on device with cache
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aclrtMallocCached interface needs to be released
 * through the aclrtFree interface.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFree | aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocCached(void **devPtr,
                                               size_t size,
                                               aclrtMemMallocPolicy policy);

/**
 * @ingroup AscendCL
 * @brief allocate device memory with config
 *
 * @param devPtr [OUT] pointer to allocated memory on device
 * @param size [IN]    alloc memory size
 * @param policy [IN]  memory alloc policy
 * @param cfg [IN]     memory alloc config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocWithCfg(void **devPtr,
                                                size_t size,
                                                aclrtMemMallocPolicy policy,
                                                aclrtMallocConfig *cfg);

/**
 * @ingroup AscendCL
 * @brief allocate device memory for task scheduler
 *
 * @param devPtr [OUT] pointer to allocated memory on device
 * @param size [IN]    alloc memory size
 * @param policy [IN]  memory alloc policy
 * @param cfg [IN]     memory alloc config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocForTaskScheduler(void **devPtr,
                                                         size_t size,
                                                         aclrtMemMallocPolicy policy,
                                                         aclrtMallocConfig *cfg);

/**
 * @ingroup AscendCL
 * @brief allocate host memory with config
 *
 * @param ptr [OUT]    pointer to allocated memory
 * @param size [IN]    alloc memory size
 * @param cfg [IN]     memory alloc config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocHostWithCfg(void **ptr,
                                                    uint64_t size,
                                                    aclrtMallocConfig *cfg);

/**
 * @ingroup AscendCL
 * @brief get memory attribute, host or device
 *
 * @param ptr [IN]         memory pointer
 * @param attributes [OUT] a buffer to store attributes
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtPointerGetAttributes(const void *ptr,
                                                       aclrtPtrAttributes *attributes);

/**
 * @ingroup AscendCL
 * @brief query the attribute of UVM memory
 *
 * @param attribute [IN]    The type of the attribute
 * @param ptr [IN]          memory pointer
 * @param size [IN]         memory size
 * @param data [OUT]        the result of the query
 * @param dataSize [IN]     the size of the buffer where the query result are stored
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemManagedGetAttr(aclrtMemManagedRangeAttribute attribute, const void *ptr, 
                                                    size_t size, void *data, size_t dataSize);

/**
 * @ingroup AscendCL
 * @brief query the attributes of UVM memory
 *
 * @param attributes [IN]        The type of the attributes
 * @param numAttributes [IN]    The number of the attributes
 * @param ptr [IN]              memory pointer
 * @param size [IN]             memory size
 * @param data [OUT]            the result of the query
 * @param dataSizes [IN]         the size of the buffer where the query results are stored
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemManagedGetAttrs(aclrtMemManagedRangeAttribute *attributes, size_t numAttributes, 
                                                     const void *ptr, size_t size, void **data, size_t *dataSizes);

/**
 * @ingroup AscendCL
 * @brief register host memory
 *
 * @param ptr [IN]     memory pointer
 * @param size [IN]    memory size
 * @param type [IN]    memory register type
 * @param ptr [OUT]    pointer to allocated memory on device
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtHostRegister(void *ptr,
                                               uint64_t size,
                                               aclrtHostRegisterType type,
                                               void **devPtr);
/**
 * @ingroup AscendCL
 * @brief register an existing host memory range
 *
 * @param ptr [IN]     host pointer to memory to page-lock
 * @param size [IN]    size in bytes of the address range to page-lock in bytes
 * @param flag [IN]    flag for allocation request
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtHostRegisterV2(void *ptr, uint64_t size, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief return device pointer of mapped host memory registered by aclrtHostRegisterV2
 *
 * @param pHost [IN]      requested host pointer mapping
 * @param pDevice [OUT]   return device pointer for mapped memory
 * @param flag [IN]       flag for extensions (must be 0 for now)
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief unregister host memory
 *
 * @param ptr [IN]     memory pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtHostUnregister(void *ptr);

/** @ingroup AscendCL
 * @brief get host mem map capabilities
 *
 * @param deviceId [IN]        device id
 * @param hacType [IN]         chip type
 * @param capabilities [OUT]   mem map capabilities
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */

ACL_FUNC_VISIBILITY aclError aclrtHostMemMapCapabilities(uint32_t deviceId, aclrtHacType hacType, aclrtHostMemMapCapability *capabilities);
/**
 * @ingroup AscendCL
 * @brief get thread last task id
 *
 * @param taskId [OUT] thread task id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetThreadLastTaskId(uint32_t *taskId);

/**
 * @ingroup AscendCL
 * @brief get stream id from a stream handle
 *
 * @param stream [IN]    stream handle
 * @param streamId [OUT] stream id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamGetId(aclrtStream stream,
                                              int32_t *streamId);

/**
 * @ingroup AscendCL
 * @brief flush cache data to ddr
 *
 * @param devPtr [IN]  the pointer that flush data to ddr
 * @param size [IN]    flush size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemFlush(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief invalidate cache data
 *
 * @param devPtr [IN]  pointer to invalidate cache data
 * @param size [IN]    invalidate size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemInvalidate(void *devPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief free device memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMalloc interface
 *
 * @param  devPtr [IN]  Pointer to memory to be freed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr);

/**
 * @ingroup AscendCL
 * @brief alloc memory on host
 *
 * @par Restriction
 * @li The requested memory cannot be used in the Device
 * and needs to be explicitly copied to the Device.
 * @li The memory requested by the aclrtMallocHost interface
 * needs to be released through the aclrtFreeHost interface.
 *
 * @param  hostPtr [OUT] pointer to pointer to allocated memory on the host
 * @param  size [IN]     alloc memory size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreeHost
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocHost(void **hostPtr, size_t size);

/**
 * @ingroup AscendCL
 * @brief alloc uvm memory
 *
 * @par Restriction
 * @li The memory requested by the aclrtMemAllocManaged interface
 * needs to be released through the aclrtFree interface.
 *
 * @param  ptr [OUT] pointer to pointer to allocated memory
 * @param  size [IN] alloc memory size
 * @param  flag [IN] flag of memory type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief free host memory
 *
 * @par Function
 *  can only free memory allocated through the aclrtMallocHost interface
 *
 * @param  hostPtr [IN]   free memory pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocHost
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeHost(void *hostPtr);

/**
 * @ingroup AscendCL
 * @brief free device memory with device synchronize
 *
 * @par Function
 *  can only free memory allocated through the aclrtMalloc interface
 *
 * @param  devPtr [IN]  Pointer to memory to be freed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeWithDevSync(void *devPtr);

/**
 * @ingroup AscendCL
 * @brief free host memory with device synchronize
 *
 * @par Function
 *  can only free memory allocated through the aclrtMallocHost interface
 *
 * @param  hostPtr [IN]   free memory pointer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocHost
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeHostWithDevSync(void *hostPtr);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param destMax [IN]   Max length of the destination address memory
 * @param src [IN]       source address pointer
 * @param count [IN]     the length of byte to copy
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief Initialize memory and set contents of memory to specified value
 *
 * @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
 * @param devPtr [IN]    Starting address of memory
 * @param maxCount [IN]  Max length of destination address memory
 * @param value [IN]     Set value
 * @param count [IN]     The length of memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device
 *
 * @par Function
 *  After calling this interface,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device, would
 *         be synchronous if memory is not allocated via calling acl or rts api.
 *
 * @par Function
 *  After calling this interface and memory is allocated via calling acl or rts api,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsyncWithCondition(void *dst,
                                                           size_t destMax,
                                                           const void *src,
                                                           size_t count,
                                                           aclrtMemcpyKind kind,
                                                           aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2d(void *dst,
                                           size_t dpitch,
                                           const void *src,
                                           size_t spitch,
                                           size_t width,
                                           size_t height,
                                           aclrtMemcpyKind kind);

/**
 * @ingroup AscendCL
 * @brief asynchronous memory replication of two-dimensional matrix between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param dpitch [IN]    pitch of destination memory
 * @param src [IN]       source address pointer
 * @param spitch [IN]    pitch of source memory
 * @param width [IN]     width of matrix transfer
 * @param height [IN]    height of matrix transfer
 * @param kind [IN]      memcpy type
 * @param stream [IN]    asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy2dAsync(void *dst,
                                                size_t dpitch,
                                                const void *src,
                                                size_t spitch,
                                                size_t width,
                                                size_t height,
                                                aclrtMemcpyKind kind,
                                                aclrtStream stream);

/**
* @ingroup AscendCL
* @brief Asynchronous initialize memory
* and set contents of memory to specified value async
*
* @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
* @param devPtr [IN]      destination address pointer
* @param maxCount [IN]    Max length of destination address memory
* @param value [IN]       set value
* @param count [IN]       the number of byte to set
* @param stream [IN]      asynchronized task stream
*
* @retval ACL_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*
* @see aclrtSynchronizeStream
*/
ACL_FUNC_VISIBILITY aclError aclrtMemsetAsync(void *devPtr,
                                              size_t maxCount,
                                              int32_t value,
                                              size_t count,
                                              aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Set/cancel the properties of a section of UVM memory
 *
 * @param ptr [IN]       destination address pointer
 * @param size [IN]      the number of byte to set
 * @param advise [IN]    advise type
 * @param location [IN]  the location information of physical memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemManagedAdvise(const void *const ptr, uint64_t size,
                                                    aclrtMemManagedAdviseType advise, aclrtMemManagedLocation location);
/**
 * @ingroup AscendCL
 * @brief Allocate an address range reservation
 *
 * @param virPtr [OUT]    Resulting pointer to start of virtual address range allocated
 * @param size [IN]       Size of the reserved virtual address range requested
 * @param alignment [IN]  Alignment of the reserved virtual address range requested
 * @param expectPtr [IN]  Fixed starting address range requested, must be nullptr
 * @param flags [IN]      Flag of page type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReleaseMemAddress | aclrtMallocPhysical | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(void **virPtr,
                                                    size_t size,
                                                    size_t alignment,
                                                    void *expectPtr,
                                                    uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Free an address range reservation
 *
 * @param virPtr [IN]  Starting address of the virtual address range to free
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReserveMemAddress
 */
ACL_FUNC_VISIBILITY aclError aclrtReleaseMemAddress(void *virPtr);

/**
 * @ingroup AscendCL
 * @brief Create a memory handle representing a memory allocation of a given
 * size described by the given properties
 *
 * @param handle [OUT]  Value of handle returned. All operations on this
 * allocation are to be performed using this handle.
 * @param size [IN]     Size of the allocation requested
 * @param prop [IN]     Properties of the allocation to create
 * @param flags [IN]    Currently unused, must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreePhysical | aclrtReserveMemAddress | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle,
                                                 size_t size,
                                                 const aclrtPhysicalMemProp *prop,
                                                 uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Release a memory handle representing a memory allocation which was
 * previously allocated through aclrtMallocPhysical
 *
 * @param handle [IN]  Value of handle which was returned previously by aclrtMallocPhysical
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMallocPhysical
 */
ACL_FUNC_VISIBILITY aclError aclrtFreePhysical(aclrtDrvMemHandle handle);

/**
 * @ingroup AscendCL
 * @brief Maps an allocation handle to a reserved virtual address range
 *
 * @param virPtr [IN]  Address where memory will be mapped
 * @param size [IN]    Size of the memory mapping
 * @param offset [IN]  Offset into the memory represented by handle from which to start mapping
 * @param handle [IN]  Handle to a shareable memory
 * @param flags [IN]   Currently unused, must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtUnmapMem | aclrtReserveMemAddress | aclrtMallocPhysical
 */
ACL_FUNC_VISIBILITY aclError aclrtMapMem(void *virPtr,
                                         size_t size,
                                         size_t offset,
                                         aclrtDrvMemHandle handle,
                                         uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Unmap the backing memory of a given address range
 *
 * @param virPtr [IN]  Starting address for the virtual address range to unmap
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtUnmapMem(void *virPtr);

/**
 * @ingroup AscendCL
 * @brief Create config handle of stream
 *
 * @retval the aclrtStreamConfigHandle pointer
 */
ACL_FUNC_VISIBILITY aclrtStreamConfigHandle *aclrtCreateStreamConfigHandle(void);

/**
 * @ingroup AscendCL
 * @brief Destroy config handle of model execute
 *
 * @param  handle [IN]  Pointer to aclrtStreamConfigHandle to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamConfigHandle(aclrtStreamConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief set config for stream
 *
 * @param handle [OUT]    pointer to stream config handle
 * @param attr [IN]       config attr in stream config handle to be set
 * @param attrValue [IN]  pointer to stream config value
 * @param valueSize [IN]  memory size of attrValue
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamConfigOpt(aclrtStreamConfigHandle *handle, aclrtStreamConfigAttr attr,
    const void *attrValue, size_t valueSize);

/**
 * @ingroup AscendCL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream);

/**
 * @ingroup AscendCL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 * @param  handle [IN]   the config of stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStreamV2(aclrtStream *stream, const aclrtStreamConfigHandle *handle);

/**
 * @ingroup AscendCL
 * @brief  create stream instance with param
 *
 * @par Function
 * Can create fast streams through the aclrtCreateStreamWithConfig interface
 *
 * @param  stream [OUT]   the created stream
 * @param  priority [IN]   the priority of stream, value range:0~7
 * @param  flag [IN]   indicate the function for stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @par Restriction
 * Before calling the aclrtDestroyStream interface to destroy
 * the specified Stream, you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the Stream have been completed.
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream | aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief destroy stream instance by force
 *
 * @par Function
 * Can only destroy streams created through the aclrtCreateStream interface
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateStream
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamForce(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 * -1 means waiting indefinitely, 0 means check whether synchronization is complete immediately
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Query a stream for completion status.
 *
 * @param  stream [IN]   the stream to query
 * @param  status [OUT]  stream status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status);

/**
 * @ingroup AscendCL
 * @brief Query priority of the stream .
 *
 * @param  stream [IN]   the stream to query
 * @param  priority [OUT]  stream priority
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamGetPriority(aclrtStream stream, uint32_t *priority);

/**
 * @ingroup AscendCL
 * @brief Query flags of the stream .
 *
 * @param  stream [IN]   the stream to query
 * @param  flags [OUT]  stream flags
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamGetFlags(aclrtStream stream, uint32_t *flags);

/**
 * @ingroup AscendCL
 * @brief Blocks the operation of the specified Stream until
 * the specified Event is completed.
 * Support for multiple streams waiting for the same event.
 *
 * @param  stream [IN]   the wait stream If using thedefault Stream, set NULL
 * @param  event [IN]    the event to wait
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);

/**
 * @ingroup AscendCL
 * @brief Blocks the operation of the specified Stream until
 * the specified Event is completed or the timeout period elapses.
 * Support for multiple streams waiting for the same event.
 * Returns an error code if the wait times out.
 *
 * @param  stream [IN]   the wait stream If using the default Stream, set NULL
 * @param  event [IN]    the event to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEventWithTimeout(aclrtStream stream, aclrtEvent event, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief set group
 *
 * @par Function
 *  set the task to the corresponding group
 *
 * @param groupId [IN]   group id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo | aclrtGetGroupInfoDetail
 */
ACL_FUNC_VISIBILITY aclError aclrtSetGroup(int32_t groupId);

/**
 * @ingroup AscendCL
 * @brief get the number of group
 *
 * @par Function
 *  get the number of group. if the number of group is zero,
 *  it means that group is not supported or group is not created.
 *
 * @param count [OUT]   the number of group
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupCount(uint32_t *count);

/**
 * @ingroup AscendCL
 * @brief create group information
 *
 * @retval null for failed.
 * @retval OtherValues success.
 *
 * @see aclrtDestroyGroupInfo
 */
ACL_FUNC_VISIBILITY aclrtGroupInfo *aclrtCreateGroupInfo();

/**
 * @ingroup AscendCL
 * @brief destroy group information
 *
 * @param groupInfo [IN]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtCreateGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get all group information
 *
 * @param groupInfo [OUT]   pointer to group information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount
 */
ACL_FUNC_VISIBILITY aclError aclrtGetAllGroupInfo(aclrtGroupInfo *groupInfo);

/**
 * @ingroup AscendCL
 * @brief get detail information of group
 *
 * @param groupInfo [IN]    pointer to group information
 * @param groupIndex [IN]   group index value
 * @param attr [IN]         group attribute
 * @param attrValue [OUT]   pointer to attribute value
 * @param valueLen [IN]     length of attribute value
 * @param paramRetSize [OUT]   pointer to real length of attribute value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetGroupCount | aclrtGetAllGroupInfo
 */
ACL_FUNC_VISIBILITY aclError aclrtGetGroupInfoDetail(const aclrtGroupInfo *groupInfo,
                                                     int32_t groupIndex,
                                                     aclrtGroupAttr attr,
                                                     void *attrValue,
                                                     size_t valueLen,
                                                     size_t *paramRetSize);

/**
 * @ingroup AscendCL
 * @brief checking whether current device and peer device support the p2p feature
 *
 * @param canAccessPeer [OUT]   pointer to save the checking result
 * @param deviceId [IN]         current device id
 * @param peerDeviceId [IN]     peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceEnablePeerAccess | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief enable the peer device to support the p2p feature
 *
 * @param peerDeviceId [IN]   the peer device id
 * @param flags [IN]   reserved field, now it must be zero
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceDisablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);

/**
 * @ingroup AscendCL
 * @brief disable the peer device to support the p2p function
 *
 * @param peerDeviceId [IN]   the peer device id
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtDeviceCanAccessPeer | aclrtDeviceEnablePeerAccess
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId);

/**
 * @ingroup AscendCL
 * @brief Obtain the free memory and total memory of specified attribute.
 * the specified memory include normal memory and huge memory.
 *
 * @param attr [IN]    the memory attribute of specified device
 * @param free [OUT]   the free memory of specified device
 * @param total [OUT]  the total memory of specified device.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);

/**
 * @ingroup AscendCL
 * @brief Query the device memory information occupied by each component.
 *
 * @param deviceId [IN]             the deviceId to be queried.
 * @param memUsageInfo [IN/OUT]     the memUsageInfo used to store memory usage information.
 * @param inputNum [IN]             the number of components that are expected to be queried.
 * @param outputNum [IN/OUT]        the actual number of components queried.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemUsageInfo(int32_t deviceId, aclrtMemUsageInfo *memUsageInfo, size_t inputNum, size_t *outputNum);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for waitting of op
 *
 * @param timeout [IN]   op wait timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpWaitTimeout(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for op executing
 *
 * @param timeout [IN]   op execute timeout, the unit is seconds
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for op executing
 *
 * @param timeout [IN]   op execute timeout, the unit is milliseconds
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOutWithMs(uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for op executing and get actual effective timeout
 *
 * @param timeout [IN]   op execute timeout, the unit is microseconds
 * @param actualTimeout [OUT]   op actual execute timeout, the unit is microseconds
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t *actualTimeout);


/**
 * @ingroup AscendCL
 * @brief Get the smallest timeout interval for op executing
 *
 * @param interval [OUT]   op actual execute timeout, the unit is microseconds
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetOpTimeOutInterval(uint64_t *interval);

/**
 * @ingroup AscendCL
 * @brief enable or disable overflow switch on some stream
 * @param stream [IN]   set overflow switch on this stream
 * @param flag [IN]  0 : disable 1 : enable
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief get overflow switch on some stream
 * @param stream [IN]   get overflow switch on this stream
 * @param flag [OUT]  current overflow switch, 0 : disable others : enable
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag);

/**
 * @ingroup AscendCL
 * @brief set saturation mode
 * @param mode [IN]   target saturation mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode);

/**
 * @ingroup AscendCL
 * @brief get saturation mode
 * @param mode [OUT]   get saturation mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode *mode);

/**
 * @ingroup AscendCL
 * @brief get overflow status asynchronously
 *
 * @par Restriction
 * After calling the aclrtGetOverflowStatus interface,
 * you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the stream have been completed.
 * @param outputAddr [IN/OUT]  output device addr to store overflow status
 * @param outputSize [IN]  output addr size
 * @param outputSize [IN]  stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetOverflowStatus(void *outputAddr, size_t outputSize, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief reset overflow status asynchronously
 *
 * @par Restriction
 * After calling the aclrtResetOverflowStatus interface,
 * you need to call the aclrtSynchronizeStream interface
 * to ensure that the tasks in the stream have been completed.
 * @param outputSize [IN]  stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetOverflowStatus(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief The thread that handles the hostFunc function on the Stream
 *
 * @param hostFuncThreadId [IN] thread ID
 * @param exeStream        [IN] stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream);

/**
 * @ingroup AscendCL
 * @brief After waiting for a specified time, trigger hostFunc callback function processing
 *
 * @par Function
 *  The thread processing callback specified by the aclrtSubscribeHostFunc interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSubscribeHostFunc
 */
ACL_FUNC_VISIBILITY aclError aclrtProcessHostFunc(int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief Cancel thread registration,
 *        the hostFunc function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param hostFuncThreadId [IN]   thread ID
 * @param exeStream        [IN]   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream);

/**
 * @ingroup AscendCL
 * @brief Get device status
 *
 * @param deviceId       [IN]   device ID
 * @param deviceStatus   [OUT]  device status
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtQueryDeviceStatus(int32_t deviceId, aclrtDeviceStatus *deviceStatus);

/**
 * @ingroup AscendCL
 * @brief Create data of type aclrtBinary
 *
 * @param [in] data   binary data
 * @param [in] dataLen   binary length
 *
 * @retval the aclrtBinary
 */
ACL_FUNC_VISIBILITY aclrtBinary aclrtCreateBinary(const void *data, size_t dataLen);

/**
 * @ingroup AscendCL
 * @brief Destroy data of type aclrtBinary
 *
 * @param modelDesc [IN]   aclrtBinary to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyBinary(aclrtBinary binary);


/**
 * @ingroup AscendCL
 * @brief Registers and parses the bin file and loads it to the device.
 *
 * @param [in] binary   device binary description
 * @param [out] binHandle   device binary handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryLoad(const aclrtBinary binary, aclrtBinHandle *binHandle);


/**
 * @ingroup AscendCL
 * @brief UnLoad binary
 *
 * @param [in] binHandle  binary handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle);

/**
 * @ingroup AscendCL
 * @brief Find funcHandle based on binHandle and kernel name
 *
 * @param [in] binHandle  binHandle
 * @param [in] kernelName   kernel name
 * @param [out] funcHandle   funcHandle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
                                                    aclrtFuncHandle *funcHandle);

/**
 * @ingroup AscendCL
 * @brief Kernel Launch to device
 * @param [in] funcHandle  function handle
 * @param [in] numBlocks  block dimentions
 * @param [in] argsData  args data
 * @param [in] argsSize  args size
 * @param [in] stream   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchKernel(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                               const void *argsData, size_t argsSize, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief share the handle that created by the process itself to other process
 * @param [in] handle       mem handle created by aclrtMallocPhysical
 * @param [in] handleType  reserved param, must be MEM_HANDLE_TYPE_NONE
 * @param [in] flags       flags for this operation. Valid flags are:
 *                           ACL_RT_VMM_EXPORT_FLAG_DEFAULT : Default behavior.
 *                           ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION : Remove whitelist verification for PID.
 * @param [out]            shareableHandle  shareable Handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle,
                                                             aclrtMemHandleType handleType, uint64_t flags,
                                                             uint64_t *shareableHandle);

 /**
 * @ingroup AscendCL
 * @brief share the handle that created by the process itself to other process
 * @param [in] handle   mem handle created by aclrtMallocPhysical
 * @param [in] flags  reserved param, must be 0
 * @param [in] shareType  share type of shareableHandle
 * @param [out] shareableHandle  shareable Handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemExportToShareableHandleV2(aclrtDrvMemHandle handle, uint64_t flags, 
    aclrtMemSharedHandleType shareType, void *shareableHandle);                                                            

/**
 * @ingroup AscendCL
 * @brief import a mem allocation from a shareable Handle
 * @param [in] shareableHandle  shareable Handle
 * @param [in] deviceId  used to generate the handle in the specified Device Id
 * @param [out] handle handle in the process
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle,
                                                               int32_t deviceId, aclrtDrvMemHandle *handle);

/**
 * @ingroup AscendCL
 * @brief import a mem allocation from a shareable Handle
 * @param [in] shareableHandle  shareable Handle
 * @param [in] shareType  share type of shareableHandle
 * @param [in] flags  reserved param, must be 0
 * @param [out] handle handle in the process
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemImportFromShareableHandleV2(void *shareableHandle, 
    aclrtMemSharedHandleType shareType, uint64_t flags, aclrtDrvMemHandle *handle);

/**
 * @ingroup AscendCL
 * @brief set the process whitelist, only the process configured in the whitelist can use this shareableHandle
 * @param [in] shareableHandle  shareable Handle
 * @param [in] deviceId  used to generate the handle in the specified Device Id
 * @param [out] handle handle in the process
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle,
                                                             int32_t *pid, size_t pidNum);

/**
 * @ingroup AscendCL
 * @brief set the process whitelist, only the process configured in the whitelist can use this shareableHandle
 * @param [in] shareableHandle  shareable Handle
 * @param [in] shareType  share type of shareableHandle
 * @param [in] pid  array for storing trustlisted process IDs
 * @param [in] pidNum number of processes in the trustlist
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemSetPidToShareableHandleV2(void *shareableHandle, 
    aclrtMemSharedHandleType shareType, int32_t *pid, size_t pidNum);

/**`
 * @ingroup AscendCL
 * @brief get the mem allocation granularity by the option
 * @param [in] prop  aclrtPhysicalMemProp
 * @param [in] option  mem granularity option
 * @param [out] granularity granularity
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp *prop,
                                                              aclrtMemGranularityOptions option,
                                                              size_t *granularity);

/**
 * @ingroup AscendCL
 * @brief Get the pid for the current process on the physical device
 * @param [out] pid value of pid
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetBareTgid(int32_t *pid);

/**
 * @ingroup AscendCL
 * @brief cache manager operation
 * @param [in] src  device memory address
 * @param [in] size  memory size
 * @param [in] cmoType  type of operation
 * @param [in] stream   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCmoAsync(void *src, size_t size, aclrtCmoType cmoType, aclrtStream stream);

/**`
 * @ingroup AscendCL
 * @brief get the mem uce info
 * @param [in] deviceId
 * @param [in/out] memUceInfoArray
 * @param [in] arraySize
 * @param [out] retSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo *memUceInfoArray,
                                                size_t arraySize, size_t *retSize);

/**`
 * @ingroup AscendCL
 * @brief stop the task on specified device
 * @param [in] deviceId
 * @param [in] timeout
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceTaskAbort(int32_t deviceId, uint32_t timeout);

/**`
 * @ingroup AscendCL
 * @brief repair the mem uce
 * @param [in] deviceId
 * @param [in/out] memUceInfoArray
 * @param [in] arraySize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo *memUceInfoArray, size_t arraySize);

/**`
 * @ingroup AscendCL
 * @brief abort unexecuted tasks and pause executing tasks on the stream
 * @param [in] stream  stream to be aborted, cannot be null
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamAbort(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Load kernel binary from file with given path
 * @param [in] binPath
 * @param [in] options
 * @param [out] binHandle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
                                                     aclrtBinHandle *binHandle);
/**
 * @ingroup AscendCL
 * @brief Get Bin dev address
 * @param [in] binHandle  bin handle
 * @param [out] binAddr  bin address
 * @param [out] binSize  bin size
 * @retval ACL_SUCCESS the function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryGetDevAddress(const aclrtBinHandle binHandle, void **binAddr, size_t *binSize);

/**
 * @ingroup AscendCL
 * @brief Get function handle with the function entry
 * @param [in] binHandle
 * @param [in] funcEntry
 * @param [out] funcHandle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry,
                                                           aclrtFuncHandle *funcHandle);
/**
 * @ingroup AscendCL
 * @brief Get kernel pc start address in device
 * @param [in] funcHandle
 * @param [out] aicAddr
 * @param [out] aivAddr
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void **aicAddr, void **aivAddr);

/**
 * @ingroup AscendCL
 * @brief Get kernel size
 * @param [in] funcHandle function handle
 * @param [out] aicSize size of AI Cube kernel
 * @param [out] aivSize size of AI Vector kernel
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetFunctionSize(aclrtFuncHandle funcHandle, size_t *aicSize, size_t *aivSize);

/**
 * @ingroup AscendCL
 * @brief Get memcpy desc size
 * @param [in] kind
 * @param [out] descSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetMemcpyDescSize(aclrtMemcpyKind kind, size_t *descSize);

/**
 * @ingroup AscendCL
 * @brief Set memcpy desc
 * @param [out] desc
 * @param [in] kind
 * @param [in] srcAddr
 * @param [in] dstAddr
 * @param [in] count
 * @param [in] config
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetMemcpyDesc(void *desc, aclrtMemcpyKind kind, void *srcAddr, void *dstAddr,
                                                size_t count, void *config);

/**
 * @ingroup AscendCL
 * @brief Asynchronous memcpy with  offset
 * @param [in] dst destination address pointer
 * @param [in] destMax length of destination memory
 * @param [in] dstDataOffset destination data address offset
 * @param [in] src source address pointer
 * @param [in] count the number of byte to copy
 * @param [in] dstDataOffset source data address offset
 * @param [in] kind memcpy type
 * @param [in] stream asynchronous task stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsyncWithOffset(void **dst, size_t destMax, size_t dstDataOffset, const void **src,
    size_t count, size_t srcDataOffset, aclrtMemcpyKind kind, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Use desc to do memcpy
 * @param [in] desc
 * @param [in] kind
 * @param [in] stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsyncWithDesc(void *desc, aclrtMemcpyKind kind, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Get kernel args handle memory size
 * @param [in] funcHandle
 * @param [out] memSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHandle, size_t *memSize);

/**
 * @ingroup AscendCL
 * @brief Get kernel args memory size
 * @param [in] funcHandle
 * @param [in] userArgsSize
 * @param [out] actualArgsSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHandle, size_t userArgsSize,
                                                       size_t *actualArgsSize);

/**
 * @ingroup AscendCL
 * @brief Initial kernel args
 * @param [in] funcHandle
 * @param [out] argsHandle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle);

/**
 * @ingroup AscendCL
 * @brief Initialize kernel args by user memory
 * @param [in] funcHandle
 * @param [out] argsHandle
 * @param [in] userHostMem
 * @param [in] actualArgsSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsInitByUserMem(aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle,
                                                          void *userHostMem, size_t actualArgsSize);

/**
 * @ingroup AscendCL
 * @brief Append kernel args
 * @param [in] argsHandle
 * @param [in] param
 * @param [in] paramSize
 * @param [out] paramHandle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
                                                   aclrtParamHandle *paramHandle);

/**
 * @ingroup AscendCL
 * @brief Append kernel args placeholder
 * @param [in] argsHandle
 * @param [out] paramHandle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle,
                                                              aclrtParamHandle *paramHandle);

/**
 * @ingroup AscendCL
 * @brief Get kernel args buffer of placeholder
 * @param [in] argsHandle
 * @param [in] paramHandle
 * @param [in] dataSize
 * @param [out] bufferAddr
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHandle,
                                                                 aclrtParamHandle paramHandle, size_t dataSize,
                                                                 void **bufferAddr);

/**
 * @ingroup AscendCL
 * @brief Update kernel args placeholder
 * @param [in] argsHandle
 * @param [in] paramHandle
 * @param [in] param
 * @param [in] paramSize
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsParaUpdate(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle,
                                                       void *param, size_t paramSize);

/**
 * @ingroup AscendCL
 * @brief Launch kernel
 * @param [in] funcHandle
 * @param [in] numBlocks
 * @param [in] stream
 * @param [in] cfg
 * @param [in] argsHandle
 * @param [in] reserve
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                                         aclrtStream stream, aclrtLaunchKernelCfg *cfg,
                                                         aclrtArgsHandle argsHandle, void *reserve);

/**
 * @ingroup AscendCL
 * @brief Get task parameters
 * @details Retrieve current parameter information from the specified task
 * @note  This API only supports AclGraph
 * @param task [in] task handle
 * @param params [out] Output parameter to store the retrieved parameter information
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRITaskGetParams(aclmdlRITask task, aclmdlRITaskParams* params);

/**
 * @ingroup AscendCL
 * @brief Set task parameters
 * @details Update parameter information for the specified task
 * @note  This API only supports AclGraph 
 * @param task [in] task handle
 * @param params [in] Input parameter containing parameter information to be set
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRITaskSetParams(aclmdlRITask task, aclmdlRITaskParams* params);

/**
 * @ingroup AscendCL
 * @brief Get kernel task launch config info
 * @details Get the launch config info for the specified kernel task
 * @note  This API only supports AclGraph 
 * @param task [in] task handle
 * @param attrId [in] the id for config info
 * @param attrValue [out] Output config value corresponding to the attrId
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIKernelTaskGetAttribute(aclmdlRITask task, aclrtLaunchKernelAttrId attrId, aclrtLaunchKernelAttrValue *attrValue);

/**
 * @ingroup AscendCL
 * @brief update model
 * @param modelRI [in] model runtime instance
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIUpdate(aclmdlRI modelRI);

/**
 * @ingroup AscendCL
 * @brief Set the task update flag to disabled
 * @param [in] task  task handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRITaskDisable(aclmdlRITask task);

/**
 * @ingroup AscendCL
 * @brief Finalize kernel args
 * @param [in] argsHandle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle);

/**
 * @ingroup AscendCL
 * @brief mem write value
 * @param [in] devAddr  dev addr
 * @param [in] value    write value
 * @param [in] flag     reserved, must be 0
 * @param [in] stream   asynchronized task stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtValueWrite(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief mem wait value
 * @param [in] devAddr  dev addr
 * @param [in] value    expect value
 * @param [in] flag     wait mode
 * @param [in] stream   asynchronized task stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtValueWait(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief get the number of available streams.
 * @param [out] streamCount   the number of available streams currently
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamAvailableNum(uint32_t *streamCount);

/**
 * @ingroup AscendCL
 * @brief set stream attribute
 * @param [in] stream       stream handle
 * @param [in] stmAttrType  stream attribute type, which value can be:
 *                             ACL_STREAM_ATTR_FAILURE_MODE, ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK
 *                             or ACL_STREAM_ATTR_USER_CUSTOM_TAG, ACL_STREAM_ATTR_CACHE_OP_INFO
 * @param [in] value        stream attribute value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType,
    aclrtStreamAttrValue *value);

/**
 * @ingroup AscendCL
 * @brief get stream attribute
 * @param [in] stream       stream handle
 * @param [in] stmAttrType  stream attribute type, which value can be:
 *                             ACL_STREAM_ATTR_FAILURE_MODE, ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK
 *                             or ACL_STREAM_ATTR_USER_CUSTOM_TAG, ACL_STREAM_ATTR_CACHE_OP_INFO
 * @param [out] value       stream attribute value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType,
    aclrtStreamAttrValue *value);

/**
 * @ingroup AscendCL
 * @brief create a notify
 * @param [out] notify  notify to be created
 * @param [in] flag     notify flag, reserved, must be 0
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateNotify(aclrtNotify *notify, uint64_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy a notify
 * @param [in] notify  notify to be destroy
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyNotify(aclrtNotify notify);

/**
 * @ingroup AscendCL
 * @brief create a count notify
 * @param [out] cntNotify  cntNotify to be created
 * @param [in] flag        cntNotify flag, reserved, must be 0
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCntNotifyCreate(aclrtCntNotify *cntNotify, uint64_t flag);

/**
 * @ingroup AscendCL
 * @brief destroy a notify
 * @param [in] cntNotify  cntNotify to be destroy
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCntNotifyDestroy(aclrtCntNotify cntNotify);

/**
 * @ingroup AscendCL
 * @brief record a notify
 * @param [in] notify  notify to be recorded
 * @param [in] stream  input stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief wait for a notify with timeout
 * @param [in] notify   notify to be wait
 * @param [in] stream   input stream
 * @param [in] timeout  input timeout, unit is ms
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout);

/**
 * @ingroup AscendCL
 * @brief get notify id
 * @param [in] notify     notify to be get
 * @param [out] notifyId  notify id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId);

/**
 * @ingroup AscendCL
 * @brief get event id
 * @param [in] event     event to be get
 * @param [out] eventId  event id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId);

/**
 * @ingroup AscendCL
 * @brief get avaliable event count
 * @param [out] eventCount  avaliable event count
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetEventAvailNum(uint32_t *eventCount);

/**
 * @ingroup AscendCL
 * @brief get device infomation.
 * @param [in] deviceId  the device id
 * @param [in] attr      device attr
 * @param [out] value    the device info
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value);

/**
 * @ingroup AscendCL
 * @brief get priority range of current device
 * @param [out] leastPriority     least priority
 * @param [out] greatestPriority  greatest priority
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority);

/**
 * @ingroup AscendCL
 * @brief get device feature ability by device id, such as task schedule ability.
 * @param [in] deviceId        device id
 * @param [in] devFeatureType  device feature type
 * @param [out] value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCapability(int32_t deviceId, aclrtDevFeatureType devFeatureType,
    int32_t *value);

/**
 * @ingroup AscendCL
 * @brief get uuid of device by device id
 * @param [in] deviceId        device id
 * @param [out] uuid           16-byte Universally Unique Identifier for 
 *                              globally unique identification of an NPU device.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetUuid(int32_t deviceId, aclrtUuid *uuid);

/**
 * @ingroup AscendCL
 * @brief get h2d atomic capabilities by device id
 * @param [out] capabilities  atomic capabilities
 * @param [in] operations     atomic operations
 * @param [in] count          atomic operations count
 * @param [in] deviceId       device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetHostAtomicCapabilities(
    uint32_t* capabilities, const aclrtAtomicOperation* operations, const uint32_t count, int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief get p2p atomic capabilities by device id
 * @param [out] capabilities  atomic capabilities
 * @param [in] operations     atomic operations
 * @param [in] count          atomic operations count
 * @param [in] srcDeviceId    source device id
 * @param [in] dstDeviceId    destination device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDeviceGetP2PAtomicCapabilities(
    uint32_t* capabilities, const aclrtAtomicOperation* operations, const uint32_t count, int32_t srcDeviceId,
    int32_t dstDeviceId);

/**
 * @ingroup AscendCL
 * @brief get current default stream
 * @param [out] stream  default stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream);

/**
 * @ingroup AscendCL
 * @brief get the state of primary context
 * @param [in] deviceId        device id
 * @param [out] flags  reserved, must be nullptr
 * @param [out] active state of primary context, 0 is inactive, 1 is active
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetPrimaryCtxState(int32_t deviceId, uint32_t *flags, int32_t *active);

/**
 * @ingroup AscendCL
 * @brief get current default stream
 * @param [in] dst      dst memory
 * @param [in] src      src memory
 * @param [in] count    count
 * @param [in] kind     reduce kind
 * @param [in] type     data type
 * @param [in] stream   associated stream
 * @param [in] reserve  reserved, must be nullptr
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtReduceAsync(void *dst, const void *src, uint64_t count, aclrtReduceKind kind,
    aclDataType type, aclrtStream stream, void *reserve);

/**
 * @ingroup AscendCL
 * @brief Get the value of the current device's limited resources
 * @param [in] deviceId  the device id
 * @param [in] type      resources type
 * @param [out] value    resources limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t *value);

/**
 * @ingroup AscendCL
 * @brief Set the value of the current device's limited resources
 * @param [in] deviceId  the device id
 * @param [in] type      resource type
 * @param [in] value     resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value);

/**
 * @ingroup AscendCL
 * @brief Reset the value of the current device's limited resources
 * @param [in] deviceId  the device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDeviceResLimit(int32_t deviceId);

/**
 * @ingroup AscendCL
 * @brief Get the value of the limited resources of the specified stream
 * @param [in] stream   the stream handle
 * @param [in] type     resource type
 * @param [out] value   resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t *value);

/**
 * @ingroup AscendCL
 * @brief Set the value of the limited resources of the specified stream
 * @param [in] stream   the stream handle
 * @param [in] type     resource type
 * @param [in] value    resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t value);

/**
 * @ingroup AscendCL
 * @brief Reset the value of the limited resources of the specified stream
 * @param [in] stream   the stream handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetStreamResLimit(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Use stream resource in current thread
 * @param stream [in]  stream to use
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUseStreamResInCurrentThread(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Not use stream resource in current thread
 * @param stream [in]  stream to not use
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtUnuseStreamResInCurrentThread(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Get the value of the limited resources of the current thread
 * @param type [in]   resource type
 * @param value [out] resource limit value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t *value);

/**
 * @ingroup AscendCL
 * @brief create label instance
 * @param label [out]  created label
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateLabel(aclrtLabel *label);

/**
 * @ingroup AscendCL
 * @brief set label and stream instance
 * @param label [in]  set label
 * @param stream [in] stream to be set
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetLabel(aclrtLabel label, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief destroy label instance
 * @param label [in] label to destroy
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyLabel(aclrtLabel label);

/**
 * @ingroup AscendCL
 * @brief create label list
 * @param labels [in]     model label list
 * @param num [in]        label number
 * @param labelList [out] created label list
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCreateLabelList(aclrtLabel *labels, size_t num, aclrtLabelList *labelList);

/**
 * @ingroup AscendCL
 * @brief destory label list
 * @param labelList [in]  label list to destroy
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDestroyLabelList(aclrtLabelList labelList);

/**
 * @ingroup AscendCL
 * @brief label switch by index
 * @param ptr [in]       index value ptr
 * @param maxValue [in]  index max value
 * @param labelList [in] label list
 * @param stream [in]    associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSwitchLabelByIndex(void *ptr, uint32_t maxValue, aclrtLabelList labelList,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief active a stream
 * @param activeStream [in] stream to be activated
 * @param stream [in]       stream to send task
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief execute extensible stream switch task
 * @param leftValue [in]   pointer of value
 * @param cond [in]        judge condition
 * @param rightValue [in]  pointer of target value
 * @param dataType [in]    data type of target value
 * @param trueStream [in]  stream to be activated when condition is met
 * @param falseStream [in] reserved parameter
 * @param stream [in]      stream to send task
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSwitchStream(void *leftValue, aclrtCondition cond, void *rightValue,
    aclrtCompareDataType dataType, aclrtStream trueStream, aclrtStream falseStream, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief get kernel name
 * @param funcHandle [in] function Handle
 * @param maxLen [in]     max length of kernel name
 * @param name [out]      kernel name
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name);

/**
 * @ingroup AscendCL
 * @brief get aclrtMbuf from aclrtMbuf chain by index
 * @param headBuf [in]  aclrtMbuf chain head
 * @param index [in]    the index which is smaller than num acquired from aclrtGetBufChainNum
 * @param buf [out]     the aclrtMbuf from aclrtMbuf on index
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAppendBufChain aclrtGetBufChainNum
 */
ACL_FUNC_VISIBILITY aclError aclrtGetBufFromChain(aclrtMbuf headBuf, uint32_t index, aclrtMbuf *buf);

/**
 * @ingroup AscendCL
 * @brief get aclrtMbuf chain total size
 * @param headBuf [in]  aclrtMbuf chain head
 * @param num [out]     aclrtMbuf chain total size
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAppendBufChain
 */
ACL_FUNC_VISIBILITY aclError aclrtGetBufChainNum(aclrtMbuf headBuf, uint32_t *num);

/**
 * @ingroup AscendCL
 * @brief append aclrtMbuf to aclrtMbuf chain
 * @param headBuf [in]  aclrtMbuf chain head
 * @param buf [in]      aclrtMbuf to be appended
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtAppendBufChain(aclrtMbuf headBuf, aclrtMbuf buf);

/**
 * @ingroup AscendCL
 * @brief copy buf ref
 * @param buf [in]      aclrtMbuf
 * @param newBuf [out]  Make a reference copy of the data area of buf and
 *                      create a new buf header pointing to the same data area
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCopyBufRef(const aclrtMbuf buf, aclrtMbuf *newBuf);

/**
 * @ingroup AscendCL
 * @brief get private data buf address and size
 * @param buf [in]       aclrtMbuf
 * @param dataPtr [out]  pointer to the user ptr
 * @param size [in]      the current private data area size, less than or equal to 96B
 * @param offset [in]    address offset, less than or equal to 96B
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetBufUserData(const aclrtMbuf buf, void *dataPtr, size_t size, size_t offset);

/**
 * @ingroup AscendCL
 * @brief set private data buf address and size
 * @param buf [out]     aclrtMbuf
 * @param dataPtr [in]  pointer to the user ptr
 * @param size [in]     the current private data area size, less than or equal to 96B
 * @param offset [in]   address offset, less than or equal to 96B
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetBufUserData(aclrtMbuf buf, const void *dataPtr, size_t size, size_t offset);

/**
 * @ingroup AscendCL
 * @brief get data buf address
 * @param buf [in]       aclrtMbuf
 * @param dataPtr [out]  pointer to the data ptr which is acquired from aclrtMbuf
 * @param size [out]     pointer to the size
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAllocBuf
 */
ACL_FUNC_VISIBILITY aclError aclrtGetBufData(const aclrtMbuf buf, void **dataPtr, size_t *size);

/**
 * @ingroup AscendCL
 * @brief get data buf effective len
 * @param buf [in]   aclrtMbuf
 * @param len [out]  get effective len which is set by aclrtSetBufDataLen
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSetBufDataLen
 */
ACL_FUNC_VISIBILITY aclError aclrtGetBufDataLen(aclrtMbuf buf, size_t *len);

/**
 * @ingroup AscendCL
 * @brief set data buf effective len
 * @param buf [in]  aclrtMbuf
 * @param len [in]  set effective len to data buf which must be smaller than size acquired by aclrtGetBufData
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtGetBufData aclrtGetBufDataLen
 */
ACL_FUNC_VISIBILITY aclError aclrtSetBufDataLen(aclrtMbuf buf, size_t len);

/**
 * @ingroup AscendCL
 * @brief free aclrtMbuf
 * @param buf [in]  pointer to the aclrtMbuf
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtAllocBuf
 */
ACL_FUNC_VISIBILITY aclError aclrtFreeBuf(aclrtMbuf buf);

/**
 * @ingroup AscendCL
 * @brief alloc aclrtMbuf
 * @param buf [out]  pointer to the aclrtMbuf
 * @param size [in]  size of aclrtMbuf
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtFreeBuf
 */
ACL_FUNC_VISIBILITY aclError aclrtAllocBuf(aclrtMbuf *buf, size_t size);

/**
 * @ingroup AscendCL
 * @brief Load kernel binary from given bin data buffer
 * @param data [in]         kernel binary data ptr
 * @param length [in]       kernel binary data length
 * @param options [in]      optional, can be nullptr
 * @param binHandle [out]   load result
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtBinaryLoadFromData(const void *data, size_t length,
    const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle);

/**
 * @ingroup AscendCL
 * @brief register cpu func by funcName and kernelName, funcHandle will get
 * @param handle [in]       binary handle to register func
 * @param funcName [in]     cpu kernel func name
 * @param kernelName [in]   cpu kernel op type
 * @param funcHandle [out]  func Handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRegisterCpuFunc(const aclrtBinHandle handle, const char *funcName,
    const char *kernelName, aclrtFuncHandle *funcHandle);

/**
 * @ingroup AscendCL
 * @brief launch common cmo task on the stream.
 * @param src [in]       prefetch addrs
 * @param size [in]      prefetch addrs load
 * @param cmoType [in]   opcode
 * @param barrierId [in] logic barrier Id. >0 valid Id, =0 invalid id
 * @param stream [in]    associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCmoAsyncWithBarrier(void *src, size_t size, aclrtCmoType cmoType,
    uint32_t barrierId, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief launch barrier cmo task on the stream.
 * @param taskInfo [in]  barrier task info
 * @param stream [in]    launch task on the stream
 * @param flag [in]      reserved, must be 0
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCmoWaitBarrier(aclrtBarrierTaskInfo *taskInfo, aclrtStream stream, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief get devices topo info
 * @param deviceId [in]       the logical device id
 * @param otherDeviceId [in]  the other logical device id
 * @param value [out]         topo info
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetDevicesTopo(uint32_t deviceId, uint32_t otherDeviceId, uint64_t *value);

/**
 * @ingroup AscendCL
 * @brief Perform a batch of memory copies synchronous.
 * @param dsts [in]          dest pointers.
 * @param destMaxs [in]       array of destination address memory max length
 * @param srcs [in]          src pointers.
 * @param sizes [in]         array of memcpy lengths.
 * @param numBatches [in]    batch number.
 * @param attrs [in]         array of memcpy attributes.
 * @param attrsIndexes [in]  attrs[n] is applied from attrsIndexes[n] to attrsIndexes[n+1] - 1. attrs[numAttrs-1]
 *                           is applied from attrsIndexes[numAttrs-1] to numBatches - 1.
 * @param numAttrs [in]      attrs and attrsIndexes number.
 * @param failIndex [out]    if all memcpy succeed or error is none memcpy related, set to SIZE_MAX.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatch(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes,
    size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, size_t *failIndex);

/**
 * @ingroup AscendCL
 * @brief Perform a batch of memory copies synchronous.
 * @param dsts [in]          dest pointers.
 * @param destMaxs [in]       array of destination address memory max length
 * @param srcs [in]          src pointers.
 * @param sizes [in]         array of memcpy lengths.
 * @param numBatches [in]    batch number.
 * @param attrs [in]         array of memcpy attributes.
 * @param attrsIndexes [in]  attrs[n] is applied from attrsIndexes[n] to attrsIndexes[n+1] - 1. attrs[numAttrs-1]
 *                           is applied from attrsIndexes[numAttrs-1] to numBatches - 1.
 * @param numAttrs [in]      attrs and attrsIndexes number.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatchV2(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes,
    size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs);

/**
 * @ingroup AscendCL
 * @brief Perform a batch of memory copies asynchronous.
 * @param dsts [in]          dest pointers.
 * @param destMaxs [in]       array of destination address memory max length
 * @param srcs [in]          src pointers.
 * @param sizes [in]         array of memcpy lengths.
 * @param numBatches [in]    batch number.
 * @param attrs [in]         array of memcpy attributes.
 * @param attrsIndexes [in]  attrs[n] is applied from attrsIndexes[n] to attrsIndexes[n+1] - 1. attrs[numAttrs-1]
 *                           is applied from attrsIndexes[numAttrs-1] to numBatches - 1.
 * @param numAttrs [in]      attrs and attrsIndexes number.
 * @param failIndex [out]    if all memcpy succeed or error is none memcpy related, set to SIZE_MAX.
 * @param stream [IN]        asynchronized task stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 * @note if the memory is not page-locked, synchronous copying will be performed.
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatchAsync(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes,
    size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, size_t *failIndex,
    aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Perform a batch of memory copies asynchronous.
 * @param dsts [in]          dest pointers.
 * @param destMaxs [in]       array of destination address memory max length
 * @param srcs [in]          src pointers.
 * @param sizes [in]         array of memcpy lengths.
 * @param numBatches [in]    batch number.
 * @param attrs [in]         array of memcpy attributes.
 * @param attrsIndexes [in]  attrs[n] is applied from attrsIndexes[n] to attrsIndexes[n+1] - 1. attrs[numAttrs-1]
 *                           is applied from attrsIndexes[numAttrs-1] to numBatches - 1.
 * @param numAttrs [in]      attrs and attrsIndexes number.
 * @param stream [IN]        asynchronized task stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 * @note if the memory is not page-locked, synchronous copying will be performed.
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyBatchAsyncV2(void **dsts, size_t *destMaxs, void **srcs, size_t *sizes,
    size_t numBatches, aclrtMemcpyBatchAttr *attrs, size_t *attrsIndexes, size_t numAttrs, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief make memory shared interprocess and get key
 * @param devPtr [in]  device memory address pointer
 * @param size [in]    identification byteCount
 * @param key [out]    identification key
 * @param len [in]     key length
 * @param flags [in]   flags for this operation. Valid flags are:
 *                       ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT : Default behavior.
 *                       ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION : Remove whitelist verification for PID.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief destroy a interprocess shared memory
 * @param key [in]  identification key
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcMemClose(const char *key);

/**
 * @ingroup AscendCL
 * @brief open a interprocess shared memory
 * @param devPtr [out]  pointer to device memory pointer
 * @param key [in]      identification key
 * @param flags [in]    flags for this operation. Valid flags are:
 *                        ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT : Default behavior.
 *                        ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS : Enables direct access to memory allocations on a peer device.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Ipc set mem pid
 * @param key [in]  key to be queried
 * @param pid [in]  process id
 * @param num [in]  length of pid
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num);

/**
 * @ingroup AscendCL
 * @brief Set the attribute of shared memory
 * @param key [in]   identification key
 * @param type [in]  memory attribute type
 * @param attr [in]  attribute value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcMemSetAttr(const char *key, aclrtIpcMemAttrType type, uint64_t attr);

/**
 * @ingroup AscendCL
 * @brief Ipc set mem pid
 * @param key [in]  identification key
 * @param serverPids [in]  whitelisted server pids
 * @param num [in]  length of serverPids
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcMemImportPidInterServer(const char *key, aclrtServerPid *serverPids, size_t num);

/**
 * @ingroup AscendCL
 * @brief Batch reset notify
 * @param notifies [in]  notify to be reset
 * @param num [in]       length of notifies
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNotifyBatchReset(aclrtNotify *notifies, size_t num);

/**
 * @ingroup AscendCL
 * @brief Set a notify to IPC notify
 * @param notify [in]  notify to be set to IPC notify
 * @param key [out]    identification key
 * @param len [in]     length of key
 * @param flags [in]   flags for this operation. Valid flags are:
 *                       ACL_RT_NOTIFY_EXPORT_FLAG_DEFAULT : Default behavior.
 *                       ACL_RT_NOTIFY_EXPORT_FLAG_DISABLE_PID_VALIDATION : Remove whitelist verification for PID.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNotifyGetExportKey(aclrtNotify notify, char *key, size_t len, uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Open IPC notify
 * @param notify [out]  pointer to opened notify
 * @param key [in]      identification key
 * @param flags [in]    flags for this operation. Valid flags are:
 *                        ACL_RT_NOTIFY_IMPORT_FLAG_DEFAULT : Default behavior.
 *                        ACL_RT_NOTIFY_IMPORT_FLAG_ENABLE_PEER_ACCESS : Enables direct access to notify allocations on a peer device
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNotifyImportByKey(aclrtNotify *notify, const char *key, uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Ipc set notify pid
 * @param notify [in]  notify to be queried
 * @param pid [in]     process id
 * @param num [in]     length of pid
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNotifySetImportPid(aclrtNotify notify, int32_t *pid, size_t num);

/**
 * @ingroup AscendCL
 * @brief Set the server pids of the shared notify
 * @param notify [in]  notify to be queried
 * @param serverPids [in]  whitelisted server pids
 * @param num [in]     length of serverPids
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNotifySetImportPidInterServer(aclrtNotify notify, aclrtServerPid *serverPids, size_t num);

/**
 * @ingroup AscendCL
 * @brief begin capture
 * @param stream [IN] set the stream to be captured
 * @param mode [IN] capture mode
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode);

/**
 * @ingroup AscendCL
 * @brief obtain the capture information of a stream
 * @param stream [IN] stream to be queried
 * @param status [OUT] return the stream status
 * @param modelRI [OUT] return the model runtime instance
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status,
                                                    aclmdlRI *modelRI);

/**
 * @ingroup AscendCL
 * @brief end the stream capture and obtain the corresponding model
 * @param stream [IN] stream to be ended
 * @param modelRI [OUT] return the model runtime instance
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureEnd(aclrtStream stream, aclmdlRI *modelRI);

/**
 * @ingroup AscendCL
 * @brief print model information
 * @param modelRI [IN] model runtime instance
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclmdlRIDebugPrint is deprecated, use aclmdlRIDebugJsonPrint instead")
ACL_FUNC_VISIBILITY aclError aclmdlRIDebugPrint(aclmdlRI modelRI);

/**
 * @ingroup AscendCL
 * @brief print model information
 * @param modelRI [IN] model runtime instance
 * @param path [IN] json file path
 * @param flags [IN] flags for print mode
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIDebugJsonPrint(aclmdlRI modelRI, const char *path, uint32_t flags);

/**
 * @ingroup AscendCL
 * @brief exchange capture mode
 * @param mode [IN/OUT] in the current thread, use the input mode to swap out the current mode
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode);

/**
 * @ingroup AscendCL
 * @brief Execute model asynchronous inference until the inference result is returned
 *
 * @param  modelRI [IN]   runtime instance of the model to perform inference
 * @param  stream [IN]    stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief execute model instance synchronously
 * @param [in] modelRI  model to execute, timeout for sync
 * @param [in] timeout  max waiting duration of synchronization, unit is ms
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIExecute(aclmdlRI modelRI, int32_t timeout);

/**
 * @ingroup AscendCL
 * @brief destroy the model
 *
 * @param  modelRI [IN]   runtime instance of the model to be destroyed
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIDestroy(aclmdlRI modelRI);

/**
 * @ingroup AscendCL
 * @brief the start interface of the task group
 * @param stream [IN] capture stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureTaskGrpBegin(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief the end interface of the task group
 * @param stream [IN] capture stream
 * @param handle [OUT] task group handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureTaskGrpEnd(aclrtStream stream, aclrtTaskGrp *handle);

/**
 * @ingroup AscendCL
 * @brief begin to update the task group specified by the handle
 * @param stream [IN] specify the stream used for task update
 * @param handle [IN] task group handle
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureTaskUpdateBegin(aclrtStream stream, aclrtTaskGrp handle);

/**
 * @ingroup AscendCL
 * @brief end the update of the task
 * @param stream [IN] specify the stream used for task update
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRICaptureTaskUpdateEnd(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief create model
 * @param [in] modelRI  model
 * @param [in] flag     reserved, must be 0
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIBuildBegin(aclmdlRI *modelRI, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief bind model and stream instance
 * @param [in] modelRI  binded model
 * @param [in] stream   binded stream
 * @param [in] flag     flag, which value can be ACL_MODEL_STREAM_FLAG_HEAD or ACL_MODEL_STREAM_FLAG_DEFAULT
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIBindStream(aclmdlRI modelRI, aclrtStream stream, uint32_t flag);

/**
 * @ingroup AscendCL
 * @brief add a end graph task to stream
 * @param [in] modelRI  model to execute
 * @param [in] stream   graph stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIEndTask(aclmdlRI modelRI, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief tell runtime Model has been Loaded
 * @param [in] modelRI  model to execute
 * @param [in] reserve  reserved, must be nullptr
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIBuildEnd(aclmdlRI modelRI, void *reserve);

/**
 * @ingroup AscendCL
 * @brief unbind model and stream instance
 * @param [in] modelRI  unbinded model
 * @param [in] stream   unbinded stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIUnbindStream(aclmdlRI modelRI, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief set model name
 * @param [in] modelRI  model to execute
 * @param [in] name     model name
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRISetName(aclmdlRI modelRI, const char *name);

/**
 * @ingroup AscendCL
 * @brief get model name
 * @param [in] modelRI  model to execute
 * @param [in] maxLen   max length of model name
 * @param [out] name    model name
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIGetName(aclmdlRI modelRI, uint32_t maxLen, char *name);

/**
 * @ingroup AscendCL
 * @brief register callback func for model destroy
 * @param [in] modelRI  model to execute
 * @param [in] func     callback func
 * @param [in] ptr      User data to be passed to the callback function
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIDestroyRegisterCallback(aclmdlRI modelRI, aclrtCallback func, void *userData);

/**
 * @ingroup AscendCL
 * @brief unregister callback func for model destroy
 * @param [in] modelRI  model to execute
 * @param [in] func       callback func
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIDestroyUnregisterCallback(aclmdlRI modelRI, aclrtCallback func);

/**
 * @ingroup AscendCL
 * @brief init dump
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlInitDump();

/**
 * @ingroup AscendCL
 * @brief set param of dump
 *
 * @param dumpCfgPath [IN]   the path of dump config
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlSetDump(const char *dumpCfgPath);

/**
 * @ingroup AscendCL
 * @brief finalize dump.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclmdlFinalizeDump();

/**
 * @ingroup AscendCL
 * @brief get streams from the model
 * @param [in] modelRI: model handle
 * @param [in, out] streams: array to store the retrieved streams
 * @param [in] numStreams: size of streams array
 * @param [out] numStreams: actual number of streams retrieved
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIGetStreams(aclmdlRI modelRI, aclrtStream *streams, uint32_t *numStreams);

/**
 * @ingroup AscendCL
 * @brief get tasks from the model stream
 * @param [in] stream: model stream handle
 * @param [in, out] tasks: array to store the retrieved task
 * @param [in] numTasks: size of tasks array
 * @param [out] numTasks: actual number of tasks retrieved
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIGetTasksByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks);

/**
 * @ingroup AscendCL
 * @brief get the type of the task
 * @param [in] task: task handle
 * @param [in, out] type: variable to store the task type
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRITaskGetType(aclmdlRITask task, aclmdlRITaskType *type);

/**
 * @ingroup AscendCL
 * @brief get sequence id of the task
 * @param [in] task: task handle
 * @param [out] id: sequence id of the task
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRITaskGetSeqId(aclmdlRITask task, uint32_t *id);

/**
 * @ingroup AscendCL
 * @brief query ACL interface version
 *
 * @param majorVersion[OUT] ACL interface major version
 * @param minorVersion[OUT] ACL interface minor version
 * @param patchVersion[OUT] ACL interface patch version
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetVersion(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion);

/**
 * @ingroup AscendCL
 * @brief enum for  register callback type
 */
typedef enum aclRegisterCallbackType {
    ACL_REG_TYPE_ACL_MODEL,
    ACL_REG_TYPE_ACL_OP_EXECUTOR,
    ACL_REG_TYPE_ACL_OP_CBLAS,
    ACL_REG_TYPE_ACL_OP_COMPILER,
    ACL_REG_TYPE_ACL_TDT_CHANNEL,
    ACL_REG_TYPE_ACL_TDT_QUEUE,
    ACL_REG_TYPE_ACL_DVPP,
    ACL_REG_TYPE_ACL_RETR,
    ACL_REG_TYPE_OTHER = 0xFFFF,
} aclRegisterCallbackType;
typedef aclError (*aclInitCallbackFunc)(const char *configStr, size_t len, void *userData);
typedef aclError (*aclFinalizeCallbackFunc)(void *userData);

/**
 * @ingroup AscendCL
 * @brief register callback func which will be called in aclInit
 *
 * @param type [IN]  register type
 * @cbFunc cbFunc [IN]  register callback function pointer
 * @param userData[IN]  param for cbFunc when called
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclInitCallbackRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc,
                                                     void *userData);

/**
 * @ingroup AscendCL
 * @brief unregister callback func which will be called in aclInit
 *
 * @param type [IN]  register type
 * @cbFunc cbFunc [IN]  register callback function pointer
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclInitCallbackUnRegister(aclRegisterCallbackType type, aclInitCallbackFunc cbFunc);

/**
 * @ingroup AscendCL
 * @brief register callback func which will be called in aclFinalize
 *
 * @param type [IN]  register type
 * @cbFunc cbFunc [IN]  register callback function pointer
 * @param userData[IN]  param for cbFunc when called
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalizeCallbackRegister(aclRegisterCallbackType type, aclFinalizeCallbackFunc cbFunc,
                                                         void *userData);

/**
 * @ingroup AscendCL
 * @brief unregister callback func which will be called in aclFinalize
 *
 * @param type [IN]  register type
 * @cbFunc cbFunc [IN]  register callback function pointer
 * @param userData[IN]  param for cbFunc when called
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_FILE Failure
 */
ACL_FUNC_VISIBILITY aclError aclFinalizeCallbackUnRegister(aclRegisterCallbackType type,
                                                           aclFinalizeCallbackFunc cbFunc);

/**                                                     
 * @brief check memory type
 * @param [in] addrList     memory addr list
 * @param [in] size         memory addr list size
 * @param [in] memType      memory addr list type
 * @param [out] checkResult result of check memory type
 * @param [in] reserve      reserve  reserved, must be zero
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCheckMemType(void** addrList, uint32_t size, uint32_t memType, uint32_t *checkResult, uint32_t reserve);

/**
 * @ingroup AscendCL
 * @brief get logic device id by user device id
 * @param [in] userDevid     user device id
 * @param [out] logicDevId   logic device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetLogicDevIdByUserDevId(const int32_t userDevid, int32_t *const logicDevId);

/**
 * @ingroup AscendCL
 * @brief get user device id by logic device id
 * @param [in] logicDevId logic device id
 * @param [out] userDevid user device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetUserDevIdByLogicDevId(const int32_t logicDevId, int32_t *const userDevid);

/**
 * @ingroup AscendCL
 * @brief get logic device id by physical device id
 * @param [in] phyDevId    physical device id
 * @param [out] logicDevId logic device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetLogicDevIdByPhyDevId(const int32_t phyDevId, int32_t *const logicDevId);

/**
 * @ingroup AscendCL
 * @brief get logic device id by physical device id
 * @param [in] logicDevId  physical device id
 * @param [out] phyDevId   logic device id
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t *const phyDevId);

/**
 * @ingroup AscendCL
 * @brief Support users in deploying custom profiling markers at specified network locations.
 *
 * @param [in] userdata Custom information set by user for profiling markers.
 * @param [in] length   Length of userdata, currently limited to the maximum length that can carry for subsequent markers sqe.
 * @param [in] stream   Stream issued by the marker operator.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtProfTrace(void *userdata, int32_t length, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Kernel Launch to device
 * @param [in] funcHandle  function handle
 * @param [in] numBlocks  block dimentions
 * @param [in] argsData  args data
 * @param [in] argsSize  args size
 * @param [in] cfg  configuration information
 * @param [in] stream   stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchKernelV2(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                                 const void *argsData, size_t argsSize,
                                                 aclrtLaunchKernelCfg *cfg, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Launch kernel with host args
 * @param [in] funcHandle  function handle
 * @param [in] numBlocks  block dimentions
 * @param [in] stream  stream handle
 * @param [in] cfg  configuration information
 * @param [in] hostArgs  host args data
 * @param [in] argsSize  args size
 * @param [in] placeHolderArray  placeHolder array
 * @param [in] placeHolderNum  placeHolder num
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                                           aclrtStream stream, aclrtLaunchKernelCfg *cfg,
                                                           void *hostArgs, size_t argsSize,
                                                           aclrtPlaceHolderInfo *placeHolderArray,
                                                           size_t placeHolderNum);

/**
 * @ingroup AscendCL
 * @brief get context overflowAddr
 * @param [out] overflowAddr  current ctx's overflowAddr to be get
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCtxGetFloatOverflowAddr(void **overflowAddr);

/**
 * @ingroup AscendCL
 * @brief get Saturation Status task
 * @param [in] outputAddr  pointer to op output addr
 * @param [in] outputSize  op output size
 * @param [in] stream  associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetFloatOverflowStatus(void *outputAddr, uint64_t outputSize, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief clear Saturation Status task
 * @param [in] stream  associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetFloatOverflowStatus(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief launch npu get float status task
 * @param [in] outputAddr  pointer to op output addr
 * @param [in] outputSize  op output size
 * @param [in] checkMode  check mode 
 * @param [in] stream  associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNpuGetFloatOverFlowStatus(void *outputAddr, uint64_t outputSize, uint32_t checkMode, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief launch npu clear float status task
 * @param [in] checkMode  check mode
 * @param [in] stream  associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtNpuClearFloatOverFlowStatus(uint32_t checkMode, aclrtStream stream);     

/**
 * @ingroup AscendCL
 * @brief Enqueues a host function call in s strea,
 * @param [in] stream  stream handle
 * @param [in] fn      Specify the callback function to be added
 *                     The function prototype of the callback function is:
 *                     typedef void (*aclrtHostFunc)(void *args);
 * @param [in] args    User data to be passed to the callback function
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void *args);

/**
 * @ingroup AscendCL
 * @brief get kernel sync address
 * @param [out] addr  kernel sync addr
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetHardwareSyncAddr(void **addr);

/**
 * @ingroup AscendCL
 * @brief launch random num task
 * @param [in] taskInfo  random num task info
 * @param [in] stream  stream
 * @param [in] reserve  reserve param
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRandomNumAsync(const aclrtRandomNumTaskInfo *taskInfo, const aclrtStream stream, void *reserve);

/**
 * @ingroup AscendCL
 * @brief register stream state callback func
 * @param [in] regName  register name
 * @param [in] callback  callback func
 * @param [in] args  callback func args
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRegStreamStateCallback(const char *regName, aclrtStreamStateCallback callback, void *args);

/**
 * @ingroup AscendCL
 * @brief register callback func for device id by position
 * @param [in] regName  register name
 * @param [in] callback  callback func
 * @param [in] notifyPos  callback notify position
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRegDeviceStateCallback(const char *regName, aclrtDeviceStateCallback callback, void *args);

/**
 * @ingroup AscendCL
 * @brief register callback func for device task abort
 * @param [in] regName  register name
 * @param [in] callback  callback func
 * @param [in] args  callback func args
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDeviceTaskAbortCallback(const char *regName, aclrtDeviceTaskAbortCallback callback, void *args);

/**
 * @ingroup AscendCL
 * @brief get op execute task timeout time.
 * @param [out] timeoutMs  op execute timeout time
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetOpExecuteTimeout(uint32_t *const timeoutMs);

/**
 * @ingroup AscendCL
 * @brief get device p2p enable status.
 * @param [in] deviceId  the logic device id
 * @param [in] peerDeviceId  the physical device id
 * @param [in|out] status  enable status value
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtDevicePeerAccessStatus(int32_t deviceId, int32_t peerDeviceId, int32_t *status);

/**
 * @ingroup AscendCL
 * @brief stop tasks on stream
 * @param [in] stream  stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtStreamStop(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief launch update task
 * @param [in] taskStream  destination stream
 * @param [in] taskId  destination task id
 * @param [in] info  task update info
 * @param [in] execStream  associated stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtTaskUpdateAsync(aclrtStream taskStream,
                                                   uint32_t taskId,
                                                   aclrtTaskUpdateInfo *info,
                                                   aclrtStream execStream);

/**
 * @ingroup AscendCL
 * @brief get cmo desc size
 * @param size [OUT]     size cmo desc size
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCmoGetDescSize(size_t *size);

/**
 * @ingroup AscendCL
 * @brief set cmo desc
 * @param cmoDesc [IN]      cmo desc address
 * @param src [IN]      source address ptr
 * @param size [IN]       src mem Length
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCmoSetDesc(void *cmoDesc, void *src, size_t size);

/**
 * @ingroup AscendCL
 * @brief launch com addr task by com Desc
 * @param cmoDesc [IN]      cmo desc ptr
 * @param cmoType [IN]    cmo op code
 * @param stream [IN]       stream
 * @param reserve [IN]      reverse param
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCmoAsyncWithDesc(
    void *cmoDesc, aclrtCmoType cmoType, aclrtStream stream, const void *reserve);

/**
 * @ingroup AscendCL
 * @brief check socversion
 * @param socVersion [IN]   Verify Whether omSocVersion is compatible with the current device
 * @param canCompatible [OUT]   Check compatibility: return 1 for compatible, 0 for incompatible
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCheckArchCompatibility(const char *socVersion, int32_t *canCompatible);

/**
 * @ingroup AscendCL
 * @brief abort model
 * @param [in] modelRI  model to abort
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclmdlRIAbort(aclmdlRI modelRI);

/**
 * @ingroup AscendCL
 * @brief count notify record
 * @param cntNotify [in]                count notify object
 * @param stream [in]                   stream
 * @param aclrtCntNotifyRecordInfo [in] count notify record info
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCntNotifyRecord(aclrtCntNotify cntNotify, aclrtStream stream,
    aclrtCntNotifyRecordInfo *info);

/**
 * @ingroup AscendCL
 * @brief count notify wait with timeout
 * @param cntNotify [in]                count notify object
 * @param stream [in]                   stream
 * @param aclrtCntNotifyWaitInfo [in]   count notify wait info
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCntNotifyWaitWithTimeout(aclrtCntNotify cntNotify, aclrtStream stream,
    aclrtCntNotifyWaitInfo *info);

/**
 * @ingroup AscendCL
 * @brief count notify reset
 * @param cntNotify [in]    count notify object
 * @param stream [in]       stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCntNotifyReset(aclrtCntNotify cntNotify, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief get count notify id
 * @param cntNotify [in]    count notify object
 * @param stream [in]       stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtCntNotifyGetId(aclrtCntNotify cntNotify, uint32_t *notifyId);

/**
 * @ingroup AscendCL
 * @brief stream task clean
 * @param stream [in]  stream
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtPersistentTaskClean(aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Get detailed error information of the device
 *
 * @param deviceId [IN]   the device ID
 * @param errorInfo [OUT] the error information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtGetErrorVerbose(int32_t deviceId, aclrtErrorInfo *errorInfo);

/**
 * @ingroup AscendCL
 * @brief Repair device errors
 *
 * @param deviceId [IN]   the device ID
 * @param errorInfo [IN]  the error information
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtRepairError(int32_t deviceId, const aclrtErrorInfo *errorInfo);

/**
 * @ingroup AscendCL
 * @brief This command is used to set access to a reserved virtual address range for the other device.
 * @param [in] virPtr mapped address.
 * @param [in] size mapped size.
 * @param [in] desc va location and access type, when location is device, id is devid.
 * @param [in] count desc num.
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemSetAccess(void *virPtr, size_t size, aclrtMemAccessDesc *desc, size_t count);

/**
 * @ingroup AscendCL
 * @brief This command is used to get access to a reversed virtual address range for the other device.
 * @param [in] virPtr    the va that has been mapped to device memory.
 * @param [in] location  va location, when type is device, id is devid.
 * @param [in] flag      access type.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemGetAccess(void *virPtr, aclrtMemLocation *location, uint64_t *flag);

typedef enum aclrtProcessState {
    ACL_RT_PROCESS_STATE_RUNNING = 0,
    ACL_RT_PROCESS_STATE_LOCKED,
} aclrtProcessState;

/**
 * @ingroup AscendCL
 * @brief lock the NPU process which will block further aclrt api calls
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSnapShotProcessLock();

/**
 * @ingroup AscendCL
 * @brief unlock the NPU process and allow it to continue making aclrt api calls
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSnapShotProcessUnlock();

/**
 * @ingroup AscendCL
 * @brief backup the NPU process
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSnapShotProcessBackup();

/**
 * @ingroup AscendCL
 * @brief restore the NPU process from the last backup point
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
ACL_FUNC_VISIBILITY aclError aclrtSnapShotProcessRestore();

/**
 * @ingroup AscendCL
 * @brief registers a callback function for snapshot operation stages
 * @param [in] stage the snapshot stage at which the callback should be triggered
 *   The available stages are:
 *   @li ACL_RT_SNAPSHOT_LOCK_PRE         - Called before process lock for snapshot
 *   @li ACL_RT_SNAPSHOT_BACKUP_PRE       - Called before backup operation starts
 *   @li ACL_RT_SNAPSHOT_BACKUP_POST      - Called after backup operation completes
 *   @li ACL_RT_SNAPSHOT_RESTORE_PRE      - Called before restore operation starts
 *   @li ACL_RT_SNAPSHOT_RESTORE_POST     - Called after restore operation completes
 *   @li ACL_RT_SNAPSHOT_UNLOCK_POST      - Called after process unlock
 * @param [in] callback Pointer to the callback function
 * @param [in] args User-defined argument pointer passed unchanged to the callback.
 *        This can be NULL if no additional data is needed.
 * @retval ACL_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSnapShotCallbackRegister(aclrtSnapShotStage stage, aclrtSnapShotCallBack callback, void *args);

/**
 * @ingroup AscendCL
 * @brief unregisters a previously registered callback function for a snapshot stage
 * @param [in] stage the snapshot stage at which the callback should be triggered
 *   The available stages are:
 *   @li ACL_RT_SNAPSHOT_LOCK_PRE         - Called before process lock for snapshot
 *   @li ACL_RT_SNAPSHOT_BACKUP_PRE       - Called before backup operation starts
 *   @li ACL_RT_SNAPSHOT_BACKUP_POST      - Called after backup operation completes
 *   @li ACL_RT_SNAPSHOT_RESTORE_PRE      - Called before restore operation starts
 *   @li ACL_RT_SNAPSHOT_RESTORE_POST     - Called after restore operation completes
 *   @li ACL_RT_SNAPSHOT_UNLOCK_POST      - Called after process unlock
 * @param [in] callback Pointer to the callback function
 * @retval ACL_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSnapShotCallbackUnregister(aclrtSnapShotStage stage, aclrtSnapShotCallBack callback);

/**
 * @ingroup AscendCL
 * @brief cache last task shape data for profilling in aclgraph
 *
 * @param [in] infoPtr  cache data ptr
 * @param [in] infoSize cache data size
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclrtCacheLastTaskOpInfo(const void * const infoPtr, const size_t infoSize);

/**
 * @ingroup AscendCL
 * @brief cache extended info for the last task for debugging
 *
 * @param [in]  extendInfoPtr  pointer to the extended task information buffer
 * @param [in]  infoSize       size of the buffer
 *
 * @note At most 4096 bytes are cached. If infoSize is greater than 4096, only the first 4096 bytes are used.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclrtCacheLastTaskExtendInfo(const char* const extendInfoPtr, const size_t infoSize);

/**
 * @ingroup AscendCL
 * @brief get function attribute by attribute type.
 *
 * @param [in]  funcHandle function handle
 * @param [in]  attrType   function attribue type
 * @param [out] attrValue  function attribue value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclrtGetFunctionAttribute(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType, int64_t *attrValue);

/**
 * @ingroup AscendCL
 * @brief Find binHandle based on funcHandle
 *
 * @param [in] funcHandle   funcHandle
 * @param [out] binHandle   binHandle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtFunctionGetBinary(const aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle);

/**
 * @ingroup AscendCL
 * @brief get an interprocess handle for a previously allocated event.
 *
 * @param [in]  event  event allocated with ACL_EVENT_IPC flags
 * @param [out] handle handle for interprocess
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcGetEventHandle(aclrtEvent event, aclrtIpcEventHandle *handle);

/**
 * @ingroup AscendCL
 * @brief opens an interprocess event handle for user in the current process.
 *
 * @param [in]  handle  interprocess handle to open
 * @param [out] event   returns the imported event
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure.
 */
ACL_FUNC_VISIBILITY aclError aclrtIpcOpenEventHandle(aclrtIpcEventHandle handle, aclrtEvent *event);

/**
 * @ingroup AscendCL
 * @brief This command is used to return the result to the user via virtual address contrast with physical handle.
 * @param [in] virPtr   the va that has been mapped to device memory.
 * @param [out] handle  reverse lookup physical handle based on virtual address.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemRetainAllocationHandle(void* virPtr, aclrtDrvMemHandle *handle);

/**
 * @ingroup AscendCL
 * @brief This command is used to return memory properties via physical address handle.
 * @param [in] handle physical addr handle.
 * @param [out] prop prop Properties of the allocation.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemGetAllocationPropertiesFromHandle(aclrtDrvMemHandle handle, aclrtPhysicalMemProp* prop);

/**
 * @ingroup AscendCL
 * @brief Allocate an address range reservation without ucmemory
 * @param virPtr [OUT]    Resulting pointer to start of virtual address range allocated
 * @param size [IN]       Size of the reserved virtual address range requested
 * @param alignment [IN]  Alignment of the reserved virtual address range requested
 * @param expectPtr [IN]  Fixed starting address range requested, return not support if be nullptr
 * @param flags [IN]      Flag of page type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtReleaseMemAddress | aclrtMallocPhysical | aclrtMapMem
 */
ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddressNoUCMemory(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief get start address and size of memory block
 * @param ptr [IN]   Address whithin a certain memory block range
 * @param pbase [OUT]  Start address of the memory block
 * @param psize [OUT]  Size of the memory block
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemGetAddressRange(void *ptr, void **pbase, size_t *psize);

/**
 * @ingroup AscendCL
 * @brief Used for memory mapping between devices within the same process
 * @param devPtr [IN]   Device memory address
 * @param size [IN]  Size of the memory
 * @param dstDevId [IN]  Device id to which the page table mapping is to be created
 * @param flags [IN]  Reserved param
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemP2PMap(void *devPtr, size_t size, int32_t dstDevId, uint64_t flags);

/**
 * @ingroup AscendCL
 * @brief Asynchronous prefetch memory to the specified destination device.
 * @param [in] ptr      UVM(unified virtual memory) address which will be prefetched.
 * @param [in] size     size of memory in bytes.
 * @param [in] location destination physics memory location to prefetch to.
 * @param [in] flags    reserved, must be 0.
 * @param [in] stream   stream to enqueue prefetch operation.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_PARAM for error input.
 * @retval ACL_ERROR_RT_FEATURE_NOT_SUPPORT for not support feature.
 * @retval OtherValues for other failure situation.
 */
ACL_FUNC_VISIBILITY aclError aclrtMemManagedPrefetchAsync(const void* ptr, size_t size, aclrtMemManagedLocation location,
    uint32_t flags, aclrtStream stream);

/**
 * @ingroup AscendCL
 * @brief Performs a batch of memory prefetches asynchronously.
 * @param [in] ptrs            array of UVM(unified virtual memory) address which will be prefetched.
 * @param [in] sizes           array of each prefetched memory size (in byte).
 * @param [in] count           size of dptrs and sizes arrays.
 * @param [in] prefetchLocs    array of destination physics memory location to prefetch to.
 * @param [in] prefetchLocIdxs index array mapping prefetchLocs elements to a range of prefetch operations:
                               prefetchLocs[k] applies to operations from prefetchLocIdxs[k] to prefetchLocIdxs[k+1]-1;
                               prefetchLocs[numPrefetchLocs - 1] applies from prefetchLocIdxs[numPrefetchLocs-1] to count-1.
 * @param [in] numPrefetchLocs size of prefetchLocs and prefetchLocIdxs arrays.
 * @param [in] flags           reserved, must be 0.
 * @param [in] stream          stream to enqueue prefetch operation.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval ACL_ERROR_INVALID_PARAM for error input.
 * @retval ACL_ERROR_RT_FEATURE_NOT_SUPPORT for not support feature.
 * @retval OtherValues for other failure situation.
 */
ACL_FUNC_VISIBILITY aclError aclrtMemManagedPrefetchBatchAsync(const void** ptrs, size_t* sizes, size_t count,
    aclrtMemManagedLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, uint64_t flags,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_RT_H_
