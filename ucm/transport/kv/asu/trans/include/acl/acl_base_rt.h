/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_ACL_ACL_BASE_RT_H_
#define INC_EXTERNAL_ACL_ACL_BASE_RT_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

#if defined(__GNUC__) && (__GNUC__ >= 6)
#define ACL_DEPRECATED __attribute__((deprecated))
#define ACL_DEPRECATED_MESSAGE(message) __attribute__((deprecated(message)))
#elif defined(_MSC_VER)
#define ACL_DEPRECATED __declspec(deprecated)
#define ACL_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
#define ACL_DEPRECATED
#define ACL_DEPRECATED_MESSAGE(message)
#endif

typedef void *aclrtStream;
typedef void *aclmdlRITask;
typedef void *aclrtEvent;
typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtCntNotify;
typedef void *aclrtLabel;
typedef void *aclrtLabelList;
typedef void *aclrtMbuf;
typedef int aclError;
typedef uint16_t aclFloat16;
typedef struct aclDataBuffer aclDataBuffer;
typedef void *aclrtAllocatorDesc;
typedef void *aclrtAllocator;
typedef void *aclrtAllocatorBlock;
typedef void *aclrtAllocatorAddr;
typedef void *aclrtTaskGrp;

static const int ACL_ERROR_NONE = 0;
static const int ACL_SUCCESS = 0;

static const int ACL_ERROR_INVALID_PARAM = 100000;
static const int ACL_ERROR_UNINITIALIZE = 100001;
static const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
static const int ACL_ERROR_INVALID_FILE = 100003;
static const int ACL_ERROR_WRITE_FILE = 100004;
static const int ACL_ERROR_INVALID_FILE_SIZE = 100005;
static const int ACL_ERROR_PARSE_FILE = 100006;
static const int ACL_ERROR_FILE_MISSING_ATTR = 100007;
static const int ACL_ERROR_FILE_ATTR_INVALID = 100008;
static const int ACL_ERROR_INVALID_DUMP_CONFIG = 100009;
static const int ACL_ERROR_INVALID_PROFILING_CONFIG = 100010;
static const int ACL_ERROR_INVALID_MODEL_ID = 100011;
static const int ACL_ERROR_DESERIALIZE_MODEL = 100012;
static const int ACL_ERROR_PARSE_MODEL = 100013;
static const int ACL_ERROR_READ_MODEL_FAILURE = 100014;
static const int ACL_ERROR_MODEL_SIZE_INVALID = 100015;
static const int ACL_ERROR_MODEL_MISSING_ATTR = 100016;
static const int ACL_ERROR_MODEL_INPUT_NOT_MATCH = 100017;
static const int ACL_ERROR_MODEL_OUTPUT_NOT_MATCH = 100018;
static const int ACL_ERROR_MODEL_NOT_DYNAMIC = 100019;
static const int ACL_ERROR_OP_TYPE_NOT_MATCH = 100020;
static const int ACL_ERROR_OP_INPUT_NOT_MATCH = 100021;
static const int ACL_ERROR_OP_OUTPUT_NOT_MATCH = 100022;
static const int ACL_ERROR_OP_ATTR_NOT_MATCH = 100023;
static const int ACL_ERROR_OP_NOT_FOUND = 100024;
static const int ACL_ERROR_OP_LOAD_FAILED = 100025;
static const int ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026;
static const int ACL_ERROR_FORMAT_NOT_MATCH = 100027;
static const int ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED = 100028;
static const int ACL_ERROR_KERNEL_NOT_FOUND = 100029;
static const int ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED = 100030;
static const int ACL_ERROR_KERNEL_ALREADY_REGISTERED = 100031;
static const int ACL_ERROR_INVALID_QUEUE_ID = 100032;
static const int ACL_ERROR_REPEAT_SUBSCRIBE = 100033;
static const int ACL_ERROR_STREAM_NOT_SUBSCRIBE = 100034;
static const int ACL_ERROR_THREAD_NOT_SUBSCRIBE = 100035;
static const int ACL_ERROR_WAIT_CALLBACK_TIMEOUT = 100036;
static const int ACL_ERROR_REPEAT_FINALIZE = 100037;
static const int ACL_ERROR_NOT_STATIC_AIPP = 100038;
static const int ACL_ERROR_COMPILING_STUB_MODE = 100039;
static const int ACL_ERROR_GROUP_NOT_SET = 100040;
static const int ACL_ERROR_GROUP_NOT_CREATE = 100041;
static const int ACL_ERROR_PROF_ALREADY_RUN = 100042;
static const int ACL_ERROR_PROF_NOT_RUN = 100043;
static const int ACL_ERROR_DUMP_ALREADY_RUN = 100044;
static const int ACL_ERROR_DUMP_NOT_RUN = 100045;
static const int ACL_ERROR_PROF_REPEAT_SUBSCRIBE = 148046;
static const int ACL_ERROR_PROF_API_CONFLICT = 148047;
static const int ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG = 148048;
static const int ACL_ERROR_INVALID_OPP_PATH = 148049;
static const int ACL_ERROR_OP_UNSUPPORTED_DYNAMIC = 148050;
static const int ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED = 148051;
static const int ACL_ERROR_UNSUPPORTED_JPEG = 148052;
static const int ACL_ERROR_INVALID_BUNDLE_MODEL_ID = 148053;

static const int ACL_ERROR_BAD_ALLOC = 200000;
static const int ACL_ERROR_API_NOT_SUPPORT = 200001;
static const int ACL_ERROR_INVALID_DEVICE = 200002;
static const int ACL_ERROR_MEMORY_ADDRESS_UNALIGNED = 200003;
static const int ACL_ERROR_RESOURCE_NOT_MATCH = 200004;
static const int ACL_ERROR_INVALID_RESOURCE_HANDLE = 200005;
static const int ACL_ERROR_FEATURE_UNSUPPORTED = 200006;
static const int ACL_ERROR_PROF_MODULES_UNSUPPORTED = 200007;

static const int ACL_ERROR_STORAGE_OVER_LIMIT = 300000;

static const int ACL_ERROR_INTERNAL_ERROR = 500000;
static const int ACL_ERROR_FAILURE = 500001;
static const int ACL_ERROR_GE_FAILURE = 500002;
static const int ACL_ERROR_RT_FAILURE = 500003;
static const int ACL_ERROR_DRV_FAILURE = 500004;
static const int ACL_ERROR_PROFILING_FAILURE = 500005;

typedef enum {
  ACL_DT_UNDEFINED = -1,
  ACL_FLOAT = 0,
  ACL_FLOAT16 = 1,
  ACL_INT8 = 2,
  ACL_INT32 = 3,
  ACL_UINT8 = 4,
  ACL_INT16 = 6,
  ACL_UINT16 = 7,
  ACL_UINT32 = 8,
  ACL_INT64 = 9,
  ACL_UINT64 = 10,
  ACL_DOUBLE = 11,
  ACL_BOOL = 12,
  ACL_STRING = 13,
  ACL_COMPLEX64 = 16,
  ACL_COMPLEX128 = 17,
  ACL_BF16 = 27,
  ACL_INT4 = 29,
  ACL_UINT1 = 30,
  ACL_COMPLEX32 = 33,
  ACL_HIFLOAT8 = 34,
  ACL_FLOAT8_E5M2 = 35,
  ACL_FLOAT8_E4M3FN = 36,
  ACL_FLOAT8_E8M0 = 37,
  ACL_FLOAT6_E3M2 = 38,
  ACL_FLOAT6_E2M3 = 39,
  ACL_FLOAT4_E2M1 = 40,
  ACL_FLOAT4_E1M2 = 41,
} aclDataType;

typedef enum {
  ACL_FORMAT_UNDEFINED = -1,
  ACL_FORMAT_NCHW = 0,
  ACL_FORMAT_NHWC = 1,
  ACL_FORMAT_ND = 2,
  ACL_FORMAT_NC1HWC0 = 3,
  ACL_FORMAT_FRACTAL_Z = 4,
  ACL_FORMAT_NC1HWC0_C04 = 12,
  ACL_FORMAT_HWCN = 16,
  ACL_FORMAT_NDHWC = 27,
  ACL_FORMAT_FRACTAL_NZ = 29,
  ACL_FORMAT_NCDHW = 30,
  ACL_FORMAT_NDC1HWC0 = 32,
  ACL_FRACTAL_Z_3D = 33,
  ACL_FORMAT_NC = 35,
  ACL_FORMAT_NCL = 47,
  ACL_FORMAT_FRACTAL_NZ_C0_16 = 50,
  ACL_FORMAT_FRACTAL_NZ_C0_32 = 51,
  ACL_FORMAT_FRACTAL_NZ_C0_2 = 52,
  ACL_FORMAT_FRACTAL_NZ_C0_4 = 53,
  ACL_FORMAT_FRACTAL_NZ_C0_8 = 54,
} aclFormat;

typedef enum {
  ACL_DEBUG = 0,
  ACL_INFO = 1,
  ACL_WARNING = 2,
  ACL_ERROR = 3,
} aclLogLevel;

typedef enum {
  ACL_MEMTYPE_DEVICE = 0,
  ACL_MEMTYPE_HOST = 1,
  ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT = 2
} aclMemType;

typedef enum {
  ACL_OPT_DETERMINISTIC = 0,
  ACL_OPT_ENABLE_DEBUG_KERNEL = 1,
  ACL_OPT_STRONG_CONSISTENCY = 2
} aclSysParamOpt;

typedef enum {
  ACL_CANN_ATTR_UNDEFINED = -1,
  ACL_CANN_ATTR_INF_NAN = 0,
  ACL_CANN_ATTR_BF16 = 1,
  ACL_CANN_ATTR_JIT_COMPILE = 2
} aclCannAttr;

typedef enum {
  ACL_DEVICE_INFO_UNDEFINED = -1,
  ACL_DEVICE_INFO_AI_CORE_NUM = 0,
  ACL_DEVICE_INFO_VECTOR_CORE_NUM = 1,
  ACL_DEVICE_INFO_L2_SIZE = 2
} aclDeviceInfo;

/**
 * @ingroup AscendCL
 * @brief Converts data of type aclFloat16 to data of type float
 *
 * @param value [IN]   Data to be converted
 *
 * @retval Transformed data
 */
ACL_FUNC_VISIBILITY float aclFloat16ToFloat(aclFloat16 value);

/**
 * @ingroup AscendCL
 * @brief Converts data of type float to data of type aclFloat16
 *
 * @param value [IN]   Data to be converted
 *
 * @retval Transformed data
 */
ACL_FUNC_VISIBILITY aclFloat16 aclFloatToFloat16(float value);

/**
 * @ingroup AscendCL
 * @brief create data of aclDataBuffer
 *
 * @param data [IN]    pointer to data
 * @li Need to be managed by the user,
 *  call aclrtMalloc interface to apply for memory,
 *  call aclrtFree interface to release memory
 *
 * @param size [IN]    size of data in bytes
 *
 * @retval pointer to created instance. nullptr if run out of memory
 *
 * @see aclrtMalloc | aclrtFree
 */
ACL_FUNC_VISIBILITY aclDataBuffer *aclCreateDataBuffer(void *data, size_t size);

/**
 * @ingroup AscendCL
 * @brief destroy data of aclDataBuffer
 *
 * @par Function
 *  Only the aclDataBuffer type data is destroyed here.
 *  The memory of the data passed in when the aclDataDataBuffer interface
 *  is called to create aclDataBuffer type data must be released by the user
 *
 * @param  dataBuffer [IN]   pointer to the aclDataBuffer
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclCreateDataBuffer
 */
ACL_FUNC_VISIBILITY aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief update new data of aclDataBuffer
 *
 * @param dataBuffer [OUT]    pointer to aclDataBuffer
 * @li The old data need to be released by the user, otherwise it may occur memory leak leakage
 *  call aclGetDataBufferAddr interface to get old data address
 *  call aclrtFree interface to release memory
 *
 * @param data [IN]    pointer to new data
 * @li Need to be managed by the user,
 *  call aclrtMalloc interface to apply for memory,
 *  call aclrtFree interface to release memory
 *
 * @param size [IN]    size of data in bytes
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtMalloc | aclrtFree | aclGetDataBufferAddr
 */
ACL_FUNC_VISIBILITY aclError aclUpdateDataBuffer(aclDataBuffer *dataBuffer, void *data, size_t size);

/**
 * @ingroup AscendCL
 * @brief get data address from aclDataBuffer
 *
 * @param dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data address
 */
ACL_FUNC_VISIBILITY void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data size of aclDataBuffer
 *
 * @param  dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data size
 */
ACL_DEPRECATED_MESSAGE("aclGetDataBufferSize is deprecated, use aclGetDataBufferSizeV2 instead")
ACL_FUNC_VISIBILITY uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get data size of aclDataBuffer to replace aclGetDataBufferSize
 *
 * @param  dataBuffer [IN]    pointer to the data of aclDataBuffer
 *
 * @retval data size
 */
ACL_FUNC_VISIBILITY size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer);

/**
 * @ingroup AscendCL
 * @brief get size of aclDataType
 *
 * @param  dataType [IN]    aclDataType data the size to get
 *
 * @retval size of the aclDataType
 */
ACL_FUNC_VISIBILITY size_t aclDataTypeSize(aclDataType dataType);

/**
 * @ingroup AscendCL
 * @brief an interface for users to output  APP logs
 *
 * @param logLevel [IN]    the level of current log
 * @param func [IN]        the function where the log is located
 * @param file [IN]        the file where the log is located
 * @param line [IN]        Number of source lines where the log is located
 * @param fmt [IN]         the format of current log
 * @param ... [IN]         the value of current log
 */
ACL_FUNC_VISIBILITY void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line,
                                   const char *fmt, ...);

/**
 * @ingroup AscendCL
 * @brief get soc name
 *
 * @retval null for failed
 * @retval OtherValues success
*/
ACL_FUNC_VISIBILITY const char *aclrtGetSocName();

#define ACL_APP_LOG(level, fmt, ...) \
    aclAppLog(level, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

/**
 * @ingroup AscendCL
 * @brief Get a list of the available CANN attributes in current environment
 *
 * @param  cannAttrList [OUT]  list of the available CANN attributes
 * @param  num [OUT]  the number of the available CANN attributes
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues  Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetCannAttributeList(const aclCannAttr **cannAttrList, size_t *num);

/**
 * @ingroup AscendCL
 * @brief Check whether the specified CANN attribute is available in current
 * environment
 *
 * @param  cannAttr [IN]  CANN attributes to query
 * @param  num [OUT]  0/1: 0 represents unavailable , 1 available
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues  Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetCannAttribute(aclCannAttr cannAttr, int32_t *value);

/**
 * @ingroup AscendCL
 * @brief Get capability value of the specified device
 *
 * @param  deviceId [IN]  device id
 * @param  deviceInfo [IN]  device capability to query
 * @param  value [OUT]    returned device capability value
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues  Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetDeviceCapability(uint32_t deviceId, aclDeviceInfo deviceInfo, int64_t *value);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_BASE_RT_H_