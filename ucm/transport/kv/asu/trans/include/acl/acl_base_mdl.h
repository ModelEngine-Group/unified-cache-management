/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_ACL_ACL_BASE_MODEL_H_
#define INC_EXTERNAL_ACL_ACL_BASE_MODEL_H_

#include <stdint.h> 
#include <stddef.h>

#include "acl/acl_base_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_TENSOR_SHAPE_RANGE_NUM 2
#define ACL_TENSOR_VALUE_RANGE_NUM 2
#define ACL_UNKNOWN_RANK 0xFFFFFFFFFFFFFFFE

typedef struct aclTensorDesc aclTensorDesc;

// interfaces of tensor desc
/**
 * @ingroup AscendCL
 * @brief create data aclTensorDesc
 *
 * @param  dataType [IN]    Data types described by tensor
 * @param  numDims [IN]     the number of dimensions of the shape
 * @param  dims [IN]        the size of the specified dimension
 * @param  format [IN]      tensor format
 *
 * @retval aclTensorDesc pointer.
 * @retval nullptr if param is invalid or run out of memory
 */
ACL_FUNC_VISIBILITY aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format);

/**
 * @ingroup AscendCL
 * @brief destroy data aclTensorDesc
 *
 * @param desc [IN]     pointer to the data of aclTensorDesc to destroy
 */
ACL_FUNC_VISIBILITY void aclDestroyTensorDesc(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief set tensor shape range for aclTensorDesc
 *
 * @param  desc [OUT]     pointer to the data of aclTensorDesc
 * @param  dimsCount [IN]     the number of dimensions of the shape
 * @param  dimsRange [IN]     the range of dimensions of the shape
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorShapeRange(aclTensorDesc* desc,
                                                    size_t dimsCount,
                                                    int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM]);

/**
 * @ingroup AscendCL
 * @brief set value range for aclTensorDesc
 *
 * @param  desc [OUT]     pointer to the data of aclTensorDesc
 * @param  valueCount [IN]     the number of value
 * @param  valueRange [IN]     the range of value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorValueRange(aclTensorDesc* desc,
                                                    size_t valueCount,
                                                    int64_t valueRange[][ACL_TENSOR_VALUE_RANGE_NUM]);
/**
 * @ingroup AscendCL
 * @brief get data type specified by the tensor description
 *
 * @param desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval data type specified by the tensor description.
 * @retval ACL_DT_UNDEFINED if description is null
 */
ACL_FUNC_VISIBILITY aclDataType aclGetTensorDescType(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get data format specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval data format specified by the tensor description.
 * @retval ACL_FORMAT_UNDEFINED if description is null
 */
ACL_FUNC_VISIBILITY aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get tensor size specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval data size specified by the tensor description.
 * @retval 0 if description is null
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescSize(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get element count specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval element count specified by the tensor description.
 * @retval 0 if description is null
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescElementCount(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief get number of dims specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval number of dims specified by the tensor description.
 * @retval 0 if description is null
 * @retval ACL_UNKNOWN_RANK if the tensor dim is -2
 */
ACL_FUNC_VISIBILITY size_t aclGetTensorDescNumDims(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 *
 * @retval dim specified by the tensor description and index.
 * @retval -1 if description or index is invalid
 */
ACL_DEPRECATED_MESSAGE("aclGetTensorDescDim is deprecated, use aclGetTensorDescDimV2 instead")
ACL_FUNC_VISIBILITY int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index);

/**
 * @ingroup AscendCL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimSize [OUT]    size of the specified dim.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize);

/**
 * @ingroup AscendCL
 * @brief Get the range of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimRangeNum [IN]     number of dimRange.
 * @param  dimRange [OUT]       range of the specified dim.
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimRange(const aclTensorDesc *desc,
                                                      size_t index,
                                                      size_t dimRangeNum,
                                                      int64_t *dimRange);

/**
 * @ingroup AscendCL
 * @brief set tensor description name
 *
 * @param desc [OUT]       pointer to the instance of aclTensorDesc
 * @param name [IN]        tensor description name
 */
ACL_FUNC_VISIBILITY void aclSetTensorDescName(aclTensorDesc *desc, const char *name);

/**
 * @ingroup AscendCL
 * @brief get tensor description name
 *
 * @param  desc [IN]        pointer to the instance of aclTensorDesc
 *
 * @retval tensor description name.
 * @retval empty string if description is null
 */
ACL_FUNC_VISIBILITY const char *aclGetTensorDescName(aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Convert the format in the source aclTensorDesc according to
 * the specified dstFormat to generate a new target aclTensorDesc.
 * The format in the source aclTensorDesc remains unchanged.
 *
 * @param  srcDesc [IN]     pointer to the source tensor desc
 * @param  dstFormat [IN]   destination format
 * @param  dstDesc [OUT]    pointer to the pointer to the destination tensor desc
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclTransTensorDescFormat(const aclTensorDesc *srcDesc, aclFormat dstFormat,
                                                      aclTensorDesc **dstDesc);

/**
 * @ingroup AscendCL
 * @brief Set the storage format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval ACL_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclSetTensorStorageFormat is deprecated, use aclSetTensorFormat instead")
ACL_FUNC_VISIBILITY aclError aclSetTensorStorageFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the storage shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_DEPRECATED_MESSAGE("aclSetTensorStorageShape is deprecated, use aclSetTensorShape instead")
ACL_FUNC_VISIBILITY aclError aclSetTensorStorageShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief Set the format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval ACL_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief Set the original format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval ACL_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorOriginFormat(aclTensorDesc *desc, aclFormat format);

/**
 * @ingroup AscendCL
 * @brief Set the original shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorOriginShape(aclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AscendCL
 * @brief get op description info
 *
 * @param desc [IN]     pointer to tensor description
 * @param index [IN]    index of tensor
 *
 * @retval null for failed.
 * @retval OtherValues success.
 */
ACL_FUNC_VISIBILITY aclTensorDesc *aclGetTensorDescByIndex(aclTensorDesc *desc, size_t index);

/**
 * @ingroup AscendCL
 * @brief get address of tensor
 *
 * @param desc [IN]    pointer to tensor description
 *
 * @retval null for failed
 * @retval OtherValues success
 */
ACL_FUNC_VISIBILITY void *aclGetTensorDescAddress(const aclTensorDesc *desc);

/**
 * @ingroup AscendCL
 * @brief Set the dynamic input name specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  dynamicInputName [IN]       pointer to the dynamic input name
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorDynamicInput(aclTensorDesc *desc, const char *dynamicInputName);

/**
 * @ingroup AscendCL
 * @brief Set const data specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  dataBuffer [IN]       pointer to the const databuffer
 * @param  length [IN]       the length of const databuffer
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorConst(aclTensorDesc *desc, void *dataBuffer, size_t length);

/**
 * @ingroup AscendCL
 * @brief Set tensor memory type specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aclTensorDesc
 * @param  memType [IN]       ACL_MEMTYPE_DEVICE means device, ACL_MEMTYPE_HOST or
 * ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT means host
 *
 * @retval ACL_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType memType);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_BASE_MODEL_H_